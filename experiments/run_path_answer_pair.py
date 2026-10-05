"""Paired TRAIN-holdout research: relation reading and chain-locked answer rewrite.

Uses a hosted model only for diagnostics; all outputs must stay in ignored outputs/.
Gold is never read while constructing prompts or calling the model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import task4_api as api
import task4_core as core
import task4_workflow as workflow
from experiments.audit_training_data import normalized


def choose_holdout(manifest_path: Path, per_view: int = 4) -> list[dict]:
    manifest = core.read_json(manifest_path)
    held = set(manifest["splits"]["holdout"])
    samples = core.discover_samples(core.REPO_ROOT / "数据集" / "训练集", limit=0)
    old_packs = set(random.Random(20261004).sample(sorted({s.pack for s in samples}), 40))
    old_ids_path = core.REPO_ROOT / "outputs" / "deepseek_flash_train_100_research.json"
    old_ids = {row["sample_id"] for row in core.read_json(old_ids_path)} if old_ids_path.exists() else set()
    old_packs.update(s.pack for s in samples if s.sample_id in old_ids)
    for path in (core.REPO_ROOT / "outputs").glob("research*holdout*.manifest.json"):
        used_ids = {row["sample_id"] for row in core.read_json(path)}
        old_packs.update(s.pack for s in samples if s.sample_id in used_ids)
    for path in (core.REPO_ROOT / "outputs").glob("path_answer_pair*.json"):
        if ".selection." not in path.name:
            continue
        old_packs.update((core.REPO_ROOT / "数据集" / "训练集" / row["pack"])
                         for row in core.read_json(path))
    fit_qa = set()
    for pack_name in manifest["splits"]["fit"]:
        pack = core.REPO_ROOT / "数据集" / "训练集" / pack_name
        gold = {item["sample_id"]: item for item in core.read_json(
            pack / "gold" / "问答对_答案.json")}
        for item in core.read_json(pack / "问题.json"):
            fit_qa.add((normalized(item["question"]),
                        normalized(str(gold[item["sample_id"]].get("answers", "")))))
    gold_cache = {}
    def fresh(sample: core.Sample) -> bool:
        if sample.pack not in gold_cache:
            gold_cache[sample.pack] = {item["sample_id"]: item for item in
                                       core.read_json(sample.pack / "gold" / "问答对_答案.json")}
        answer = str(gold_cache[sample.pack][sample.sample_id].get("answers", ""))
        return (normalized(sample.question), normalized(answer)) not in fit_qa
    eligible = [s for s in samples if s.pack.name in held
                and s.pack not in old_packs and fresh(s)]
    random.Random(20261006).shuffle(eligible)
    kinds = ["retrospective", "prospective", "counterfactual", "unanswerable"]
    selected = []
    used = set()
    for view in "ABC":
        for kind in kinds[:per_view]:
            sample = next((s for s in eligible if s.question_type == kind
                           and s.pack not in used), None)
            if sample is None:
                raise ValueError(f"No unseen held-out pack for {view}/{kind}")
            selected.append({"sample_id": sample.sample_id, "pack": sample.pack.name,
                             "view": view, "question_type": kind})
            used.add(sample.pack)
    return selected


def response(endpoint: str, key: str, model: str, content: str,
             sample: core.Sample, allowed: set[str]) -> tuple[dict, dict]:
    usage = Counter()
    for attempt in range(2):
        raw, tokens = model_call(endpoint, key, model, content)
        usage.update({k: v for k, v in tokens.items() if isinstance(v, int)})
        try:
            return core.parse_prediction(raw, sample, allowed), dict(usage)
        except ValueError as exc:
            if attempt:
                fallback = core.parse_prediction(
                    '{"answer":"无法确定","evidence_chain":[],"confidence":null}',
                    sample, allowed)
                return fallback, {**dict(usage), "format_fallback": True}
            content += "\n上次输出不符合要求，请只给三个字段的合法 JSON。"
    raise AssertionError("unreachable")


def model_call(endpoint: str, key: str, model: str, content: str) -> tuple[str, dict]:
    spent = Counter()
    for cap in (8192, 16384, 32768):
        spent["request_count"] += 1
        try:
            raw, usage = api.call_model(endpoint, key, model, content,
                                        max_tokens=cap, return_usage=True)
            spent.update({k: v for k, v in usage.items() if isinstance(v, int)})
            return raw, dict(spent)
        except api.TruncatedResponseError as exc:
            spent.update({k: v for k, v in exc.usage.items() if isinstance(v, int)})
            spent["truncated_calls"] += 1
            if cap == 32768:
                raise
    raise AssertionError("unreachable")


def rewrite(endpoint: str, key: str, model: str, original: str,
            first: dict, sample: core.Sample, allowed: set[str]) -> tuple[dict, dict]:
    if first["answer"] == "无法确定":
        return first, {"skipped_refusal": True}
    instruction = (original + "\n\n已锁定的证据链："
                   + json.dumps(first["evidence_chain"], ensure_ascii=False)
                   + "\n第一版答案：" + first["answer"]
                   + "\n请根据原文逐条核对答案所述事实、因果方向、问题限定和缺失的关键内容。"
                     "只可修改答案文字，不能修改、扩展或缩短已锁定证据链。"
                     "不要加入材料外事实。只输出 JSON 对象 {\"answer\":\"...\"}。")
    raw, usage = model_call(endpoint, key, model, instruction)
    try:
        answer = json.loads(raw)["answer"]
        if not isinstance(answer, str) or not answer.strip() or answer.strip() == "无法确定":
            raise ValueError("empty answer or refusal conflicts with locked chain")
        candidate = dict(first)
        candidate["answer"] = answer.strip()
        parsed = core.parse_prediction(json.dumps(candidate, ensure_ascii=False),
                                       sample, allowed)
        if parsed["evidence_chain"] != first["evidence_chain"]:
            raise ValueError("locked chain changed")
        return parsed, usage
    except (ValueError, KeyError, TypeError):
        return first, {**usage, "rewrite_invalid": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=core.REPO_ROOT / "experiments" / "finetune_split_20261005.json")
    parser.add_argument("--selection", type=Path,
                        default=core.REPO_ROOT / "experiments" / "path_answer_holdout_20261005.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--api-url", default=api.DEFAULT_API_URL)
    parser.add_argument("--model", default=api.DEFAULT_MODEL)
    args = parser.parse_args()
    output = args.output.resolve()
    ignored = core.REPO_ROOT / "outputs"
    if not output.is_relative_to(ignored):
        raise ValueError("Hosted API research output must be under ignored outputs/")
    selection_path = output.with_suffix(".selection.json")
    selected = (core.read_json(args.selection) if args.selection.exists()
                else core.read_json(selection_path) if selection_path.exists()
                else choose_holdout(args.manifest))
    if selection_path.exists() and core.read_json(selection_path) != selected:
        raise ValueError("Existing output selection differs from frozen manifest")
    if not selection_path.exists():
        core.atomic_json(selection_path, selected)
    endpoint, key, model = api.read_api_settings(args)
    samples = {s.sample_id: s for s in core.discover_samples(
        core.REPO_ROOT / "数据集" / "训练集", limit=0)}
    holdout_packs = set(core.read_json(args.manifest)["splits"]["holdout"])
    if (len({row["sample_id"] for row in selected}) != len(selected)
            or len({row["pack"] for row in selected}) != len(selected)
            or any(row["pack"] not in holdout_packs or row["view"] not in "ABC"
                   or samples[row["sample_id"]].pack.name != row["pack"]
                   for row in selected)):
        raise ValueError("Selection must contain unique questions and holdout packs")
    records = core.read_json(output) if output.exists() else []
    if [r["sample_id"] for r in records] != [r["sample_id"] for r in selected[:len(records)]]:
        raise ValueError("Checkpoint order differs from selection")
    for index, spec in enumerate(selected[len(records):], start=len(records) + 1):
        sample = samples[spec["sample_id"]]
        view = spec["view"]
        original, allowed, _ = workflow.training_view_input(sample, view)
        started = time.monotonic()
        direct, direct_usage = response(endpoint, key, model, original, sample, allowed)
        if view == "A":
            guided, _, guide_meta = workflow.relation_guided_input(sample, view)
            guided_result, guided_usage = response(endpoint, key, model, guided, sample, allowed)
        else:
            guided_result, guided_usage = direct, {}
            guide_meta = {"relation_count": 0, "candidate_paths": []}
        locked, rewrite_usage = rewrite(endpoint, key, model, original,
                                        guided_result, sample, allowed)
        records.append({**spec, "model": model, "direct": direct,
                        "guided": guided_result, "locked": locked,
                        "direct_usage": direct_usage, "guided_usage": guided_usage,
                        "rewrite_usage": rewrite_usage,
                        "direct_input_chars": len(original),
                        "guided_input_chars": len(guided) if view == "A" else len(original),
                        "guide_meta": guide_meta,
                        "elapsed_seconds": round(time.monotonic() - started, 2)})
        core.atomic_json(output, records)
        print(f"[{index}/{len(selected)}] {sample.sample_id} {view} saved", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
