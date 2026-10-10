"""One-call repair diagnostic for six prior invalid guided DeepSeek outputs.

Uses the unchanged V4 system prompt and full A-view inputs. No gold is read.
This is post-hoc debugging, not a fresh estimate of method quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
from experiments.freeze_a_candidate_v4_pair_20261010 import (  # noqa: E402
    PROMPT as V4_PROMPT, sha,
)
from experiments.freeze_a_candidate_v4_repair6_20261010 import (  # noqa: E402
    OUT as MANIFEST, REPAIR_PREFIX,
)
from experiments.run_a_candidate_v4_pair_20261010 import (  # noqa: E402
    MODEL, balance, prepare,
)
from experiments.run_codex_initial_deepseek_protocol import historical_module  # noqa: E402
from experiments.run_deepseek_initial_protocol_validation import (  # noqa: E402
    call_api, read_key,
)

OUT = ROOT / "outputs" / "a_candidate_v4_repair6_20261010"
MAX_TOKENS = 32768


def user_content(row: dict, base: str, template: str) -> str:
    options = "\n".join(f"候选 {i}：{' → '.join(path)}"
                        for i, path in enumerate(row["candidate_paths"], 1))
    if template.count("{candidates}") != 1:
        raise ValueError("Repair prompt must contain one candidates marker")
    return template.replace("{candidates}", options) + base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--stop-spend", type=Decimal, default=Decimal("5.00"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or args.stop_spend <= 0:
        raise ValueError("Invalid run limits")
    frozen = core.read_json(MANIFEST)
    rows = frozen["rows"]
    if len(rows) != 6 or frozen["v4_prompt_sha256"] != sha(V4_PROMPT.read_bytes()) or (
            frozen["repair_prefix_sha256"] != sha(REPAIR_PREFIX.read_bytes())):
        raise ValueError("Frozen repair manifest or prompt changed")
    prepared = prepare(rows)
    template = REPAIR_PREFIX.read_text(encoding="utf-8")
    system = V4_PROMPT.read_text(encoding="utf-8")
    inputs = [(row, user_content(row, base, template), sample, allowed)
              for row, base, _, sample, allowed in prepared]
    signature = {"manifest_sha256": sha(MANIFEST.read_bytes()),
                 "system_sha256": sha(system), "prefix_sha256": sha(template),
                 "model": MODEL, "max_tokens": MAX_TOKENS,
                 "response_format": {"type": "json_object"},
                 "temperature": "provider default", "reasoning_effort": "provider default",
                 "thinking": "provider default", "gold_sent": False,
                 "scope": "six previously invalid guided-arm questions"}
    if args.dry_run:
        print(json.dumps({"calls": len(inputs), "total_input_chars": sum(len(x[1]) for x in inputs),
                          "max_input_chars": max(len(x[1]) for x in inputs),
                          "signature": signature}, ensure_ascii=False))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "metadata.json"
    attempts_path = OUT / "attempts.jsonl"
    key = read_key()
    now = balance(key)
    if meta_path.exists():
        meta = core.read_json(meta_path)
        if any(meta.get(name) != value for name, value in signature.items()):
            raise ValueError("Resume metadata mismatch")
        start = Decimal(meta["start_balance_cny"])
    else:
        start = now
        core.atomic_json(meta_path, {**signature, "start_balance_cny": str(start),
                                     "stop_spend_cny": str(args.stop_spend)})
    if start - now >= args.stop_spend:
        raise RuntimeError("Repair budget cap reached")
    done = {}
    if attempts_path.exists():
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
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
        return {"sample_id": row["sample_id"], "previous_failure_type": row["previous_failure_type"],
                "input_sha256": sha(content), "parsed": parsed, "attempt": attempt}

    pending = [item for item in inputs if item[0]["sample_id"] not in done]
    print(json.dumps({"started": True, "pending": len(pending),
                      "balance_cny": str(now)}, ensure_ascii=False), flush=True)
    for offset in range(0, len(pending), 3):
        batch = pending[offset:offset + 3]
        before = balance(key)
        if start - before >= args.stop_spend:
            print(json.dumps({"budget_stop": True, "balance_cny": str(before)}), flush=True)
            break
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(call, item) for item in batch]
            for future in as_completed(futures):
                record = future.result()
                with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[record["sample_id"]] = record
                print(json.dumps({"sample_id": record["sample_id"],
                                  "prior": record["previous_failure_type"],
                                  "valid": record["parsed"] is not None,
                                  "usage": record["attempt"].get("usage"),
                                  "elapsed_seconds": record["attempt"].get("elapsed_seconds"),
                                  "error": record["attempt"].get("validation_error") or
                                           record["attempt"].get("error")},
                                 ensure_ascii=False), flush=True)
        after = balance(key)
        print(json.dumps({"completed": len(done), "balance_cny": str(after),
                          "observed_spend_cny": str(start - after)}), flush=True)
    predictions = [{"view": "A", **done[row["sample_id"]]["parsed"]}
                   for row in rows if row["sample_id"] in done
                   and done[row["sample_id"]]["parsed"] is not None]
    core.atomic_json(OUT / "predictions.json", predictions)
    print(json.dumps({"finished": True, "attempts": len(done),
                      "valid": len(predictions)}), flush=True)


if __name__ == "__main__":
    main()
