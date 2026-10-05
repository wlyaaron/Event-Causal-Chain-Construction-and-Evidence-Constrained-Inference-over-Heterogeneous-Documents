"""Run the frozen A-path experiment on all blind-test questions for comparison.

Only A receives the question-conditioned path index. B/C receive the original
input and act as same-model rerun controls. Hosted API outputs stay in outputs/;
they are not training data or an official contest submission.
"""

from __future__ import annotations

import argparse
import http.client
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import task4_api as api
import task4_core as core
import task4_workflow as workflow
from run_deepseek import predict as baseline_predict
from experiments.run_path_answer_pair import response


def input_for(sample: core.Sample) -> tuple[str, set[str]]:
    original, allowed, _ = core.build_input(sample)
    if sample.track == "A":
        guided, guided_allowed, _ = workflow.causal_path_input(sample, "A")
        if guided_allowed != allowed:
            raise ValueError(f"{sample.sample_id}: path index changed allowed IDs")
        return guided, allowed
    return original, allowed


def run(dataset: Path, output: Path, key_file: Path, model: str,
        api_url: str, workers: int, limit: int) -> dict:
    ignored = (core.REPO_ROOT / "outputs").resolve()
    output = output.resolve()
    if not output.is_relative_to(ignored):
        raise ValueError("hosted API output must stay under ignored outputs/")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    samples = core.discover_samples(dataset, "all", limit)
    if any(sample.track not in {"A", "B", "C"} for sample in samples):
        raise ValueError("this research runner accepts blind-test A/B/C samples only")
    ids = [sample.sample_id for sample in samples]
    signature = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    progress = output.with_name(output.stem + ".progress.json")
    if output.exists():
        core.validate_file(output, samples, 80000)
        return {"status": "already_complete", "samples": len(samples), "output": str(output)}
    if progress.exists():
        state = core.read_json(progress)
        if state["sample_ids_sha256"] != signature or state["model"] != model:
            raise ValueError("checkpoint selection or model differs")
    else:
        state = {"model": model, "sample_ids_sha256": signature,
                 "prompt": "A:causal_path_input; B/C:core.build_input",
                 "started_utc": datetime.now(timezone.utc).isoformat(), "rows": []}
    rows = state["rows"]
    if [row["sample_id"] for row in rows] != ids[:len(rows)]:
        raise ValueError("checkpoint is not a prefix of the selected questions")
    args = argparse.Namespace(api_url=api_url, model=model, api_key_file=key_file)
    endpoint, key, selected_model = api.read_api_settings(args)
    state["wrapper"] = "A:paired_response; B/C:original_run_deepseek_predict"

    def work(sample: core.Sample) -> dict:
        content, allowed = input_for(sample)
        for attempt in range(4):
            try:
                if sample.track == "A":
                    prediction, usage = response(endpoint, key, selected_model,
                                                 content, sample, allowed)
                else:
                    prediction, _ = baseline_predict(
                        sample, endpoint, key, selected_model, 120, 2, 8192,
                        80000, True)
                    usage = {"baseline_wrapper": True}
                return {"sample_id": sample.sample_id, "track": sample.track,
                        "prediction": prediction, "usage": usage,
                        "input_chars": len(content)}
            except (RuntimeError, OSError, http.client.HTTPException,
                    json.JSONDecodeError) as exc:
                if attempt == 3:
                    raise
                print(f"{sample.sample_id}: transient {type(exc).__name__}; retry {attempt + 1}/3",
                      flush=True)
                time.sleep(10 * (attempt + 1))
        raise AssertionError("unreachable")

    start = len(rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        next_submit = start
        for index in range(start, len(samples)):
            while next_submit < min(len(samples), index + workers):
                pending[next_submit] = pool.submit(work, samples[next_submit])
                next_submit += 1
            row = pending.pop(index).result()
            rows.append(row)
            core.atomic_json(progress, state)
            if (index + 1) % 10 == 0 or index + 1 == len(samples):
                print(f"[{index + 1}/{len(samples)}] saved", flush=True)
    predictions = [row["prediction"] for row in rows]
    core.atomic_json(output, predictions)
    core.validate_file(output, samples, 80000)
    return {"status": "complete", "samples": len(samples),
            "output": str(output), "progress": str(progress)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=core.DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--model", default=api.DEFAULT_MODEL)
    parser.add_argument("--api-url", default=api.DEFAULT_API_URL)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        samples = core.discover_samples(args.dataset, "all", args.limit)
        lengths = [len(input_for(sample)[0]) for sample in samples]
        print(json.dumps({"samples": len(samples), "max_input_chars": max(lengths),
                          "total_input_chars": sum(lengths)}, ensure_ascii=False))
        return
    if args.api_key_file is None:
        parser.error("--api-key-file is required for API research")
    print(json.dumps(run(args.dataset, args.output, args.api_key_file,
                         args.model, args.api_url, args.workers, args.limit),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
