"""Exploratory 50-question DeepSeek rerun with one-chain/answer alignment.

The frozen A-view manifest and full source input are reused. Gold is never
opened or sent to the API. Because these cases informed the prompt, resulting
scores are diagnostic rather than an independent validation estimate.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import task4_core as core
from experiments.freeze_a_candidate_v4_pair_20261010 import OUT as MANIFEST, PROMPT, sha
from experiments.run_a_candidate_v4_pair_20261010 import MODEL, balance, prepare
from experiments.run_a_candidate_v4_repair6_20261010 import user_content
from experiments.run_codex_initial_deepseek_protocol import historical_module
from experiments.run_deepseek_initial_protocol_validation import call_api, read_key


ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "experiments" / "prompts" / "task4_a_candidate_answer_aligned_v2_20261010.txt"
OUT = ROOT / "outputs" / "a_candidate_answer_aligned_v2_20261010"
MAX_TOKENS = 32768


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch-questions", type=int, default=5)
    parser.add_argument("--warn-spend", type=Decimal, default=Decimal("5.00"))
    parser.add_argument("--stop-spend", type=Decimal, default=Decimal("8.00"))
    parser.add_argument("--min-balance", type=Decimal, default=Decimal("10.00"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.batch_questions <= 10:
        raise ValueError("Invalid concurrency")
    if not 0 < args.warn_spend < args.stop_spend:
        raise ValueError("Invalid budget")
    frozen = core.read_json(MANIFEST)
    if len(frozen["rows"]) != 50 or frozen["v4_prompt_sha256"] != sha(PROMPT.read_bytes()):
        raise ValueError("Frozen manifest or V4 prompt changed")
    template = PREFIX.read_text(encoding="utf-8")
    system = PROMPT.read_text(encoding="utf-8")
    inputs = [(row, user_content(row, base, template), sample, allowed)
              for row, base, _, sample, allowed in prepare(frozen["rows"])]
    signature = {"manifest_sha256": sha(MANIFEST.read_bytes()),
                 "system_sha256": sha(system), "prefix_sha256": sha(template),
                 "model": MODEL, "max_tokens": MAX_TOKENS,
                 "temperature": "provider default", "thinking": "provider default",
                 "response_format": {"type": "json_object"}, "gold_sent": False,
                 "scope": "exploratory same-50 rerun, simulated A view"}
    if args.dry_run:
        print(json.dumps({"calls": len(inputs), "total_input_chars": sum(len(x[1]) for x in inputs),
                          "max_input_chars": max(len(x[1]) for x in inputs),
                          "signature": signature}, ensure_ascii=False))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = OUT / "metadata.json"
    attempts = OUT / "attempts.jsonl"
    key = read_key()
    current = balance(key)
    if metadata.exists():
        meta = core.read_json(metadata)
        if any(meta.get(k) != v for k, v in signature.items()):
            raise ValueError("Resume signature mismatch")
        initial = Decimal(meta["start_balance_cny"])
    else:
        initial = current
        core.atomic_json(metadata, {**signature, "start_balance_cny": str(initial),
                                    "stop_spend_cny": str(args.stop_spend),
                                    "min_balance_cny": str(args.min_balance)})
    if initial - current >= args.stop_spend or current <= args.min_balance:
        raise RuntimeError("Budget cap reached before model calls")
    done = {}
    if attempts.exists():
        for line in attempts.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["sample_id"] in done:
                raise ValueError("Duplicate attempt")
            done[item["sample_id"]] = item
    old = historical_module()

    def call(item: tuple) -> dict:
        row, content, sample, allowed = item
        attempt = call_api(key, MODEL, system, content, timeout=600,
                           retries=2, max_tokens=MAX_TOKENS)
        parsed = None
        try:
            if attempt.get("raw") is None:
                raise ValueError(attempt.get("error") or "No output")
            if attempt.get("finish_reason") == "length":
                raise ValueError("Output truncated")
            original = old.Sample(sample.pack, "A", sample.sample_id,
                                  sample.question_type, sample.question)
            parsed = old.parse_prediction(attempt["raw"], original, allowed)
        except (ValueError, TypeError, KeyError) as exc:
            attempt["validation_error"] = str(exc).replace(key, "[REDACTED]")
        return {"sample_id": row["sample_id"], "input_sha256": sha(content),
                "parsed": parsed, "attempt": attempt}

    print(json.dumps({"target": len(inputs), "pending": len(inputs)-len(done),
                      "balance_cny": str(current)}, ensure_ascii=False), flush=True)
    for start in range(0, len(inputs), args.batch_questions):
        batch = [x for x in inputs[start:start+args.batch_questions]
                 if x[0]["sample_id"] not in done]
        if not batch:
            continue
        before = balance(key)
        if initial-before >= args.stop_spend or before <= args.min_balance:
            break
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(call, x) for x in batch]):
                item = future.result()
                with attempts.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                done[item["sample_id"]] = item
                print(json.dumps({"sample_id": item["sample_id"],
                                  "valid": item["parsed"] is not None,
                                  "usage": item["attempt"].get("usage"),
                                  "elapsed_seconds": item["attempt"].get("elapsed_seconds"),
                                  "error": item["attempt"].get("validation_error") or item["attempt"].get("error")},
                                 ensure_ascii=False), flush=True)
        after = balance(key)
        spent = initial-after
        print(json.dumps({"completed": len(done), "spent_cny": str(spent),
                          "balance_cny": str(after), "warn": spent >= args.warn_spend},
                         ensure_ascii=False), flush=True)
        if spent >= args.stop_spend or after <= args.min_balance:
            break
    predictions = [{"view": "A", **done[row["sample_id"]]["parsed"]}
                   for row in frozen["rows"] if row["sample_id"] in done
                   and done[row["sample_id"]]["parsed"] is not None]
    core.atomic_json(OUT / "predictions.json", predictions)
    print(json.dumps({"attempts": len(done), "valid": len(predictions)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
