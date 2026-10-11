"""Replay only test A with baseline prompt plus program candidates; reuse baseline B/C.

The archived scored-baseline prompt, original compact input, model alias and
format handling are retained. The only intended A input change is a neutral
top-five candidate preface. No gold is read or sent. Raw attempts remain in
the ignored outputs/ directory, and the final 760-record merge copies B/C
records directly from the saved 59.55 baseline file.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import task4_core as core
from experiments.run_deepseek_initial_protocol_validation import call_api, read_key
from experiments.task4_a_candidate_pool import (
    enumerate_graph_paths,
    rank_graph_paths_targeted,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "outputs/baseline_deepseek_2026-10-04"
BASELINE = ARCHIVE_DIR / "deepseek_flash_full_760.json"
RUN_START = ARCHIVE_DIR / "source_run_start_05e46ee9.zip"
RUN_RESUME = ARCHIVE_DIR / "source_resume_949e5e52.zip"
PREFIX = ROOT / "experiments/prompts/task4_original_a_candidates_only_20261011.txt"
OUT_DIR = ROOT / "outputs/task4_original_a_candidate_20261011"
BASELINE_SHA256 = "1963188881a85503aacc26078092ff470275f448fad29e27256cac22b87511e7"
SYSTEM_SHA256 = "771ead7c3c9b4a6c245727f484630fd4a153ae5defd20ae1d1be34be84e38734"


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def archived_core() -> tuple[types.ModuleType, str]:
    """Load only the source snapshot used when the scored baseline began."""
    with zipfile.ZipFile(RUN_START) as archive:
        source = archive.read("task4_core.py").decode("utf-8")
    module = types.ModuleType("task4_scored_baseline_core_05e46ee9")
    module.__file__ = str(RUN_START)
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    with zipfile.ZipFile(RUN_RESUME) as archive:
        resumed = ast.parse(archive.read("task4_core.py").decode("utf-8"))
    resume_prompt = next(ast.literal_eval(node.value) for node in resumed.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
                                 for target in node.targets))
    if sha(module.SYSTEM_PROMPT) != SYSTEM_SHA256 or resume_prompt != module.SYSTEM_PROMPT:
        raise ValueError("Archived scored-baseline prompts differ")
    return module, module.SYSTEM_PROMPT


def select_a(samples: list[core.Sample], limit: int) -> list[core.Sample]:
    selected = [sample for sample in samples if sample.track == "A"]
    if len(selected) != 210:
        raise ValueError(f"Expected 210 test-A questions, found {len(selected)}")
    if not 0 <= limit <= len(selected):
        raise ValueError("--limit must be 0..210; 0 means all A questions")
    return selected if limit == 0 else selected[:limit]


def prepare(sample: core.Sample, prefix: str,
            original_core: types.ModuleType) -> tuple[str, set[str], list[list[str]]]:
    content, allowed, _ = core.build_input(sample)
    old_sample = original_core.Sample(sample.pack, sample.track, sample.sample_id,
                                      sample.question_type, sample.question)
    historical_content, historical_allowed, _ = original_core.build_input(old_sample)
    if historical_content != content or historical_allowed != allowed:
        raise ValueError(f"Baseline input changed: {sample.sample_id}")
    documents, events, edges = core.load_pack(sample.pack)
    if events is None or edges is None:
        raise ValueError(f"A graph or event list missing: {sample.sample_id}")
    all_paths = enumerate_graph_paths(events, edges)
    candidates = [list(path) for path in rank_graph_paths_targeted(
        sample.question, sample.question_type, events, edges,
        documents, all_paths, limit=5)]
    if len(candidates) != 5 or any(node not in allowed for path in candidates for node in path):
        raise ValueError(f"Invalid A candidate pool: {sample.sample_id}")
    options = "\n".join(f"候选 {index}：{' → '.join(path)}"
                        for index, path in enumerate(candidates, 1))
    if prefix.count("{candidates}") != 1:
        raise ValueError("Candidate prefix must have exactly one marker")
    return prefix.replace("{candidates}", options) + content, allowed, candidates


def merge(a_predictions: list[dict], all_samples: list[core.Sample], output: Path) -> None:
    if sha(BASELINE.read_bytes()) != BASELINE_SHA256:
        raise ValueError("Archived 59.55 baseline file changed")
    original = core.read_json(BASELINE)
    if [row["sample_id"] for row in original] != [sample.sample_id for sample in all_samples]:
        raise ValueError("Baseline order does not match current 760 test questions")
    a_by_id = {row["sample_id"]: row for row in a_predictions}
    if len(a_by_id) != 210:
        raise ValueError("A run is incomplete")
    merged = [a_by_id[sample.sample_id] if sample.track == "A" else old
              for sample, old in zip(all_samples, original)]
    if any(new != old for sample, new, old in zip(all_samples, merged, original)
           if sample.track in {"B", "C"}):
        raise AssertionError("B/C baseline records changed")
    core.atomic_json(output, merged)
    core.validate_file(output, all_samples, 80000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="A questions to run; 0 means all 210")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be 1..8")
    all_samples = core.discover_samples(core.DEFAULT_DATASET, "all", 0)
    if len(all_samples) != 760:
        raise ValueError("Test set no longer contains 760 questions")
    samples = select_a(all_samples, args.limit)
    old, system = archived_core()
    prefix = PREFIX.read_text(encoding="utf-8")
    prepared = [(sample, *prepare(sample, prefix, old)) for sample in samples]
    signature = {"baseline_sha256": BASELINE_SHA256,
                 "system_prompt_sha256": sha(system), "candidate_prefix_sha256": sha(prefix),
                 "candidate_code_sha256": sha((ROOT / "experiments/task4_a_candidate_pool.py").read_bytes()),
                 "sample_ids_sha256": sha("\n".join(sample.sample_id for sample in samples)),
                 "user_inputs_sha256": sha("\n".join(sha(content)
                                                   for _, content, _, _ in prepared)),
                 "model": "deepseek-flash", "max_tokens_initial": 8192,
                 "max_tokens_expansion": [16384, 32768],
                 "response_format": {"type": "json_object"}, "temperature": "provider default",
                 "thinking": "provider default", "gold_sent": False,
                 "B_C_source": str(BASELINE)}
    if args.dry_run:
        print(json.dumps({"a_questions": len(samples), "a_total_chars": sum(len(x[1]) for x in prepared),
                          "a_max_chars": max(len(x[1]) for x in prepared),
                          "candidate_counts": sorted({len(x[3]) for x in prepared}),
                          "signature": signature}, ensure_ascii=False))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_a = OUT_DIR / ("A_210.json" if args.limit == 0 else f"A_{args.limit}.json")
    attempts_path = OUT_DIR / ("A_210.attempts.jsonl" if args.limit == 0
                               else f"A_{args.limit}.attempts.jsonl")
    store = core.PredictionStore(output_a, samples, signature)
    key = read_key()

    def work(item: tuple) -> tuple[dict, list[dict]]:
        sample, content, allowed, candidates = item
        historical_sample = old.Sample(sample.pack, sample.track, sample.sample_id,
                                       sample.question_type, sample.question)
        attempts = []
        request_content = content
        for correction in range(2):
            for cap in (8192, 16384, 32768):
                attempt = call_api(key, "deepseek-flash", system, request_content,
                                   timeout=600, max_tokens=cap)
                row = {"sample_id": sample.sample_id, "correction": correction,
                       "max_tokens": cap, "input_sha256": sha(request_content),
                       "candidate_paths": candidates, "attempt": attempt}
                attempts.append(row)
                if attempt.get("finish_reason") == "length":
                    if cap == 32768:
                        raise RuntimeError(f"{sample.sample_id}: output still truncated at 32768")
                    continue
                if attempt.get("raw") is None:
                    if attempt.get("http_status") in {401, 402, 403}:
                        raise RuntimeError(f"{sample.sample_id}: API rejected request")
                    raise RuntimeError(f"{sample.sample_id}: API failed: {attempt.get('error')}")
                break
            try:
                prediction = old.parse_prediction(attempt["raw"], historical_sample, allowed)
                row["parsed"] = prediction
                return prediction, attempts
            except ValueError as exc:
                row["validation_error"] = str(exc).replace(key, "[REDACTED]")
                if correction:
                    if sample.question_type != "unanswerable":
                        raise RuntimeError(f"{sample.sample_id}: original protocol leaves this invalid output unresolved") from exc
                    fallback = '{"answer":"无法确定","evidence_chain":[],"confidence":null}'
                    prediction = old.parse_prediction(fallback, historical_sample, allowed)
                    row["fallback_strict_refusal"] = True
                    return prediction, attempts
                request_content += ("\n\n上一版输出未通过提交格式校验：" + str(exc)
                                    + "。请仅输出修正后的 JSON；证据链只用上述真实事件 ID。"
                                      "无法确定且没有可核查证据时，使用精确拒答格式。")
        raise AssertionError("Unreachable")

    start = len(store.completed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        next_submit = start
        for index in range(start, len(prepared)):
            while next_submit < min(len(prepared), index + args.workers):
                pending[next_submit] = pool.submit(work, prepared[next_submit])
                next_submit += 1
            prediction, attempts = pending.pop(index).result()
            with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
                for attempt in attempts:
                    handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
            store.append(prediction)
            if (index + 1) % 20 == 0 or index + 1 == len(samples):
                print(json.dumps({"completed": index + 1, "total": len(samples)},
                                 ensure_ascii=False), flush=True)
    store.finish()
    if args.limit == 0:
        merged = OUT_DIR / "baseline_BC_candidate_A_760.json"
        merge(core.read_json(output_a), all_samples, merged)
        print(json.dumps({"merged": str(merged), "validated": 760}, ensure_ascii=False))
    else:
        print(json.dumps({"a_output": str(output_a), "validated": len(samples)},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
