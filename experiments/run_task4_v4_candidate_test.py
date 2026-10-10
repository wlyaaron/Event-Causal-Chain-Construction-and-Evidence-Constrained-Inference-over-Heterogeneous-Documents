"""Run the frozen V4 research recipe on blind-test A/B/C questions.

A receives the targeted top-five graph paths and the chain-only prefix from
the 50-question experiment. B/C receive the original V4 input. Every output
passes the same conservative refusal-format parser and structural validator.
This script never reads gold and writes only under the ignored outputs/ tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import task4_core as core
from experiments.run_deepseek_initial_protocol_validation import call_api, read_key
from experiments.task4_a_candidate_pool import (
    enumerate_graph_paths,
    rank_graph_paths_targeted,
)


ROOT = Path(__file__).resolve().parents[1]
V4_PROMPT = ROOT / "experiments/prompts/task4_merged_rules_v4_draft_20261009.txt"
A_PREFIX = ROOT / "experiments/prompts/task4_a_candidate_chain_only_20261010.txt"
DEFAULT_OUTPUT = ROOT / "outputs/task4_v4_candidate_test_smoke.json"


def digest(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def selected_samples(dataset: Path, per_track: int, all_questions: bool) -> list[core.Sample]:
    if per_track < 1:
        raise ValueError("--per-track must be positive")
    samples = core.discover_samples(dataset, "all", 0)
    if any(sample.track not in {"A", "B", "C"} for sample in samples):
        raise ValueError("Only A/B/C test folders are accepted")
    if all_questions:
        return samples
    counts: Counter[str] = Counter()
    chosen = []
    for sample in samples:
        if counts[sample.track] < per_track:
            chosen.append(sample)
            counts[sample.track] += 1
    if set(counts) != {"A", "B", "C"}:
        raise ValueError("Test subset must contain A, B and C questions")
    return chosen


def prepare_input(sample: core.Sample, prefix: str,
                  max_input_chars: int) -> tuple[str, set[str], list[list[str]]]:
    original, allowed, _ = core.build_input(sample, max_input_chars)
    payload = json.loads(original)
    if payload.pop("track") != sample.track:
        raise ValueError(f"Track mismatch: {sample.sample_id}")
    # The 50-question V4 experiments used this default JSON serialization.
    base = json.dumps(payload, ensure_ascii=False)
    if sample.track != "A":
        return base, allowed, []

    documents, events, edges = core.load_pack(sample.pack)
    if events is None or edges is None:
        raise ValueError(f"A question lacks supplied events or graph: {sample.sample_id}")
    paths = enumerate_graph_paths(events, edges)
    candidates = [list(path) for path in rank_graph_paths_targeted(
        sample.question, sample.question_type, events, edges,
        documents, paths, limit=5)]
    if any(node not in allowed for path in candidates for node in path):
        raise ValueError(f"Candidate contains an unknown ID: {sample.sample_id}")
    options = "\n".join(
        f"候选 {index}：{' → '.join(path)}"
        for index, path in enumerate(candidates, 1)
    )
    if prefix.count("{candidates}") != 1:
        raise ValueError("A prefix must contain one {candidates} marker")
    return prefix.replace("{candidates}", options) + base, allowed, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=core.DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-track", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Select the full A/B/C test set")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-input-chars", type=int, default=80000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate an existing output only")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout < 1 or args.max_input_chars < 1:
        raise ValueError("Token, timeout and input limits must be positive")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Research outputs must stay under ignored outputs/")
    if args.all and not args.dry_run and not args.validate and output == DEFAULT_OUTPUT.resolve():
        raise ValueError("--all requires an explicit --output path")

    samples = selected_samples(args.dataset, args.per_track, args.all)
    if args.validate:
        print(json.dumps({"validated": core.validate_file(
            output, samples, args.max_input_chars)}, ensure_ascii=False))
        return
    system = V4_PROMPT.read_text(encoding="utf-8")
    prefix = A_PREFIX.read_text(encoding="utf-8")
    prepared = [(sample, *prepare_input(sample, prefix, args.max_input_chars))
                for sample in samples]
    counts = Counter(sample.track for sample in samples)
    signature = {
        "dataset": str(args.dataset.resolve()),
        "sample_ids_sha256": digest("\n".join(sample.sample_id for sample in samples)),
        "user_inputs_sha256": digest("\n".join(digest(content)
                                               for _, content, _, _ in prepared)),
        "v4_prompt_sha256": digest(system), "a_prefix_sha256": digest(prefix),
        "candidate_code_sha256": digest((ROOT / "experiments/task4_a_candidate_pool.py").read_bytes()),
        "core_code_sha256": digest((ROOT / "task4_core.py").read_bytes()),
        "parser_code_sha256": digest((ROOT / "task4_refusal_format.py").read_bytes()),
        "model": args.model, "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": "provider default", "gold_sent": False,
        "route": {"A": "targeted graph top-5 + chain-only prefix + V4",
                  "B": "V4", "C": "V4"},
    }
    if args.dry_run:
        print(json.dumps({"questions": len(samples), "tracks": counts,
                          "total_input_chars": sum(len(content) for _, content, _, _ in prepared),
                          "max_input_chars": max(len(content) for _, content, _, _ in prepared),
                          "a_candidate_counts": Counter(len(paths) for sample, _, _, paths
                                                        in prepared if sample.track == "A"),
                          "signature": signature}, ensure_ascii=False))
        return

    key = read_key()
    store = core.PredictionStore(output, samples, signature, args.max_input_chars)
    attempts_path = output.with_name(output.name + ".attempts.jsonl")
    for index in range(len(store.completed), len(prepared)):
        sample, content, allowed, candidates = prepared[index]
        attempt = call_api(key, args.model, system, content,
                           timeout=args.timeout, max_tokens=args.max_tokens)
        try:
            if attempt.get("raw") is None:
                raise ValueError(attempt.get("error") or "No model output")
            if attempt.get("finish_reason") == "length":
                raise ValueError("Model output reached max_tokens")
            prediction = core.parse_prediction(attempt["raw"], sample, allowed)
            status = "valid"
        except (ValueError, TypeError, KeyError) as exc:
            prediction = None
            status = str(exc).replace(key, "[REDACTED]")
        record = {"sample_id": sample.sample_id, "track": sample.track,
                  "input_sha256": digest(content), "candidate_paths": candidates,
                  "status": status, "parsed": prediction, "attempt": attempt}
        with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if prediction is None:
            raise RuntimeError(f"{sample.sample_id}: invalid model output ({status}); "
                               "checkpoint preserved, rerun to retry this question")
        store.append(prediction)
        print(json.dumps({"progress": f"{index + 1}/{len(samples)}",
                          "sample_id": sample.sample_id, "track": sample.track,
                          "valid": True, "finish_reason": attempt.get("finish_reason"),
                          "usage": attempt.get("usage")}, ensure_ascii=False), flush=True)
    count = store.finish()
    print(json.dumps({"validated": count, "output": str(output),
                      "attempts": str(attempts_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
