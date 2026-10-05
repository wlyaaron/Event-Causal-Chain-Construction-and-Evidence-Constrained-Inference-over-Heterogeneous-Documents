"""Pair raw A-view input with question-conditioned causal paths on held-out packs.

Research API output stays under ignored outputs/. Gold is used only to exclude
train/holdout duplicate question-answer templates while selecting fresh packs;
it is never sent to the model. The frozen selection contains no answers.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import task4_api as api
import task4_core as core
import task4_workflow as workflow
from experiments.audit_training_data import normalized
from experiments.run_path_answer_pair import response


def choose_selection(manifest_path: Path, per_type: int = 3,
                     seed: int = 20261007,
                     unanswerable_count: int = 3) -> list[dict]:
    manifest = core.read_json(manifest_path)
    root = core.REPO_ROOT / "数据集" / "训练集"
    samples = core.discover_samples(root, limit=0)
    fit_qa = set()
    gold_cache = {}

    def gold(pack: Path) -> dict:
        if pack not in gold_cache:
            gold_cache[pack] = {item["sample_id"]: item for item in
                                core.read_json(pack / "gold" / "问答对_答案.json")}
        return gold_cache[pack]

    for name in manifest["splits"]["fit"]:
        pack = root / name
        for item in core.read_json(pack / "问题.json"):
            fit_qa.add((normalized(item["question"]),
                        normalized(str(gold(pack)[item["sample_id"]].get("answers", "")))))
    used_packs = set(random.Random(20261004).sample(sorted({s.pack for s in samples}), 40))
    old_100 = core.REPO_ROOT / "outputs" / "deepseek_flash_train_100_research.json"
    if old_100.exists():
        used_ids = {row["sample_id"] for row in core.read_json(old_100)}
        used_packs.update(s.pack for s in samples if s.sample_id in used_ids)
    for file in (core.REPO_ROOT / "outputs").glob("*holdout*.manifest.json"):
        used_ids = {row["sample_id"] for row in core.read_json(file)}
        used_packs.update(s.pack for s in samples if s.sample_id in used_ids)
    prior = core.REPO_ROOT / "experiments" / "path_answer_holdout_20261005.json"
    if prior.exists():
        used_packs.update(root / row["pack"] for row in core.read_json(prior))
    first_causal = core.REPO_ROOT / "experiments" / "causal_path_holdout_20261005.json"
    if first_causal.exists() and seed != 20261007:
        used_packs.update(root / row["pack"] for row in core.read_json(first_causal))
    eligible = [sample for sample in samples
                if sample.pack.name in manifest["splits"]["holdout"]
                and sample.pack not in used_packs
                and (normalized(sample.question),
                     normalized(str(gold(sample.pack)[sample.sample_id].get("answers", ""))))
                not in fit_qa]
    random.Random(seed).shuffle(eligible)
    chosen = []
    assigned = set()
    for kind in ("retrospective", "prospective", "counterfactual", "unanswerable"):
        target = unanswerable_count if kind == "unanswerable" else per_type
        candidates = [sample for sample in eligible if sample.question_type == kind]
        for sample in candidates:
            if sample.pack in assigned:
                continue
            chosen.append({"sample_id": sample.sample_id, "pack": sample.pack.name,
                           "view": "A", "question_type": kind})
            assigned.add(sample.pack)
            if sum(row["question_type"] == kind for row in chosen) == target:
                break
        if sum(row["question_type"] == kind for row in chosen) != target:
            raise ValueError(f"not enough fresh holdout packs for {kind}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=core.REPO_ROOT / "experiments" / "finetune_split_20261005.json")
    parser.add_argument("--selection", type=Path,
                        default=core.REPO_ROOT / "experiments" / "causal_path_holdout_20261005.json")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20261007,
                        help="selection seed; 20261008 reserves a second disjoint set")
    parser.add_argument("--unanswerable-count", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--api-url", default=api.DEFAULT_API_URL)
    parser.add_argument("--model", default=api.DEFAULT_MODEL)
    args = parser.parse_args()
    selected = (core.read_json(args.selection) if args.selection.exists()
                else choose_selection(args.manifest, seed=args.seed,
                                      unanswerable_count=args.unanswerable_count))
    if not args.selection.exists():
        core.atomic_json(args.selection, selected)
    if args.select_only:
        print(json.dumps({"selected": len(selected), "selection": str(args.selection)},
                         ensure_ascii=False))
        return
    if not args.output or not args.api_key_file:
        parser.error("--output and --api-key-file are required for API research")
    output = args.output.resolve()
    if not output.is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("hosted API output must stay under ignored outputs/")
    samples = {sample.sample_id: sample for sample in core.discover_samples(
        core.REPO_ROOT / "数据集" / "训练集", limit=0)}
    held = set(core.read_json(args.manifest)["splits"]["holdout"])
    if (len({row["pack"] for row in selected}) != len(selected)
            or any(row["pack"] not in held or row["view"] != "A"
                   or samples[row["sample_id"]].pack.name != row["pack"]
                   for row in selected)):
        raise ValueError("selection must use distinct held-out A-view packs")
    records = core.read_json(output) if output.exists() else []
    if [row["sample_id"] for row in records] != [row["sample_id"] for row in selected[:len(records)]]:
        raise ValueError("checkpoint differs from frozen selection")
    endpoint, key, model = api.read_api_settings(args)
    for index, spec in enumerate(selected[len(records):], start=len(records) + 1):
        sample = samples[spec["sample_id"]]
        original, allowed, _ = workflow.training_view_input(sample, "A")
        guided, _, meta = workflow.causal_path_input(sample, "A")
        started = time.monotonic()
        direct, direct_usage = response(endpoint, key, model, original, sample, allowed)
        variant, guided_usage = response(endpoint, key, model, guided, sample, allowed)
        records.append({**spec, "model": model, "direct": direct, "guided": variant,
                        "direct_usage": direct_usage, "guided_usage": guided_usage,
                        "direct_input_chars": len(original),
                        "guided_input_chars": len(guided), "guide_meta": meta,
                        "elapsed_seconds": round(time.monotonic() - started, 2)})
        core.atomic_json(output, records)
        print(f"[{index}/{len(selected)}] {sample.sample_id} saved", flush=True)


if __name__ == "__main__":
    main()
