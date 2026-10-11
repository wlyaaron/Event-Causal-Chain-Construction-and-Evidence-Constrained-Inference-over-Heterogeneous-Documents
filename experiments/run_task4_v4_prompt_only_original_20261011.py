"""Run all blind-test questions with the scored baseline protocol and V4 system prompt.

The archived baseline core, API adapter, and prediction function are loaded from
the scored run-start snapshot. The API call substitutes only the system prompt.
No candidate paths, gold answers, refusal routing, or output postprocessing are
added. Results and raw responses stay in the ignored outputs directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import types
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import task4_core as current_core
from experiments.run_deepseek_initial_protocol_validation import read_key


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/baseline_deepseek_2026-10-04/source_run_start_05e46ee9.zip"
V4_PROMPT = ROOT / "experiments/prompts/task4_merged_rules_v4_draft_20261009.txt"
OUTPUT = ROOT / "outputs/task4_v4_prompt_only_original_760_20261011.json"
EXPECTED_V4_SHA256 = "6e4d5b5f162b0505a3912ebcd71af6d99850729b522cf70fead0fe218cf2eab5"
EXPECTED_ORIGINAL_SHA256 = "771ead7c3c9b4a6c245727f484630fd4a153ae5defd20ae1d1be34be84e38734"


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def load_original() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Load the original modules without changing the live repository modules."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        sources = {name: archive.read(name + ".py") for name in
                   ("task4_core", "task4_api", "run_deepseek")}
    originals = {name: sys.modules.get(name) for name in sources}
    modules: dict[str, types.ModuleType] = {}
    try:
        for name in sources:
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / (name + ".py"))
            sys.modules[name] = module
            exec(compile(sources[name], module.__file__, "exec"), module.__dict__)
            modules[name] = module
    finally:
        for name, previous in originals.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    if digest(modules["task4_core"].SYSTEM_PROMPT) != EXPECTED_ORIGINAL_SHA256:
        raise ValueError("The archived scored-baseline system prompt changed")
    return modules["task4_core"], modules["task4_api"], modules["run_deepseek"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="First N questions for a smoke run; 0 means all 760")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be 1..8")
    if not 0 <= args.limit <= 760:
        raise ValueError("--limit must be 0..760")

    old_core, old_api, old_runner = load_original()
    system = V4_PROMPT.read_text(encoding="utf-8")
    if digest(system) != EXPECTED_V4_SHA256:
        raise ValueError("V4 prompt changed")
    all_samples = old_core.discover_samples(current_core.DEFAULT_DATASET, "all", 0)
    if len(all_samples) != 760 or Counter(s.track for s in all_samples) != Counter(A=210, B=350, C=200):
        raise ValueError("Expected A/B/C counts 210/350/200")
    samples = all_samples if args.limit == 0 else all_samples[:args.limit]
    original_inputs = []
    for sample in samples:
        old_content, old_allowed, _ = old_core.build_input(sample)
        live_sample = current_core.Sample(sample.pack, sample.track, sample.sample_id,
                                          sample.question_type, sample.question)
        live_content, live_allowed, _ = current_core.build_input(live_sample)
        if old_content != live_content or old_allowed != live_allowed:
            raise ValueError(f"Archived baseline input differs: {sample.sample_id}")
        original_inputs.append(old_content)
    signature = {
        "source_zip_sha256": digest(ARCHIVE.read_bytes()),
        "system_prompt": str(V4_PROMPT),
        "v4_prompt_sha256": digest(system),
        "original_prompt_sha256": EXPECTED_ORIGINAL_SHA256,
        "sample_ids_sha256": digest("\n".join(s.sample_id for s in samples)),
        "user_inputs_sha256": digest("\n".join(digest(x) for x in original_inputs)),
        "model": "deepseek-flash", "json_mode": True,
        "max_tokens_initial": 8192, "max_tokens_expansion": [16384, 32768],
        "temperature": "provider default", "gold_sent": False,
        "intervention": "V4 system prompt only",
    }
    if args.dry_run:
        print(json.dumps({"questions": len(samples),
                          "tracks": dict(Counter(s.track for s in samples)),
                          "max_input_chars": max(map(len, original_inputs)),
                          "signature": signature}, ensure_ascii=False))
        return

    output = OUTPUT if args.limit == 0 else OUTPUT.with_name(
        f"task4_v4_prompt_only_original_smoke_{args.limit}_20261011.json")
    attempts_path = output.with_name(output.name + ".attempts.jsonl")
    store = old_core.PredictionStore(output, samples, signature)
    api_key = read_key()
    endpoint = old_api.completion_url(old_api.DEFAULT_API_URL)
    thread_state = threading.local()
    write_lock = threading.Lock()

    def call_v4(endpoint_arg: str, key: str, model: str, content: str,
                timeout: int, retries: int, max_tokens: int,
                json_mode: bool = True) -> str:
        if endpoint_arg != endpoint or key != api_key:
            raise ValueError("Unexpected API endpoint or credential")
        details = {"sample_id": thread_state.sample_id,
                   "max_tokens": max_tokens, "input_sha256": digest(content)}
        try:
            raw = old_api.call_model(endpoint_arg, key, model, content, timeout,
                                     retries, max_tokens, json_mode=json_mode,
                                     system_prompt=system)
        except Exception as exc:
            details["error_type"] = type(exc).__name__
            details["error"] = str(exc).replace(api_key, "[REDACTED]")
            with write_lock, attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(details, ensure_ascii=False) + "\n")
            raise
        details["raw"] = raw
        with write_lock, attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(details, ensure_ascii=False) + "\n")
        return raw

    old_runner.call_model = call_v4

    def work(sample: object) -> tuple[dict, bool]:
        thread_state.sample_id = sample.sample_id
        return old_runner.predict(sample, endpoint, api_key, "deepseek-flash",
                                  600, 2, 8192, 80000, True)

    start = len(store.completed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        next_submit = start
        for index in range(start, len(samples)):
            while next_submit < min(len(samples), index + args.workers):
                pending[next_submit] = pool.submit(work, samples[next_submit])
                next_submit += 1
            record, _ = pending.pop(index).result()
            store.append(record)
            if (index + 1) % 20 == 0 or index + 1 == len(samples):
                print(json.dumps({"completed": index + 1, "total": len(samples)},
                                 ensure_ascii=False), flush=True)
    store.finish()
    old_core.validate_file(output, samples, 80000)
    current_core.validate_file(output, [current_core.Sample(
        s.pack, s.track, s.sample_id, s.question_type, s.question) for s in samples], 80000)
    print(json.dumps({"output": str(output), "validated": len(samples),
                      "sha256": digest(output.read_bytes())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
