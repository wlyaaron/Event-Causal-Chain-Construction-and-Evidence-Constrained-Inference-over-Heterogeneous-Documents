"""Blind paired DeepSeek V4 vs V4 plus five program candidates on frozen A-view 50.

No gold file is opened. Requests use the same V4 system prompt and complete
materials. The guided arm changes only a small user-message prefix.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
import task4_workflow as workflow  # noqa: E402
from experiments.freeze_a_candidate_v4_pair_20261010 import (  # noqa: E402
    OUT as MANIFEST, PROMPT, TRAIN, sha,
)
from experiments.run_codex_initial_deepseek_protocol import historical_module  # noqa: E402
from experiments.run_deepseek_initial_protocol_validation import (  # noqa: E402
    read_key, call_api,
)

OUT = ROOT / "outputs" / "a_candidate_v4_pair_holdout50_20261010"
MODEL = "deepseek-flash"
GUIDANCE = ("以下是程序依据当前材料中的有向因果图和问题生成的候选 evidence_chain，"
            "按相关性排序，仅供核对，不保证正确或完整。请以完整材料自行判断；"
            "可以选用、修正或否决候选并自行构建合法链。不要仅因候选存在就强行回答。\n"
            "候选 evidence_chain：{candidates}\n完整原始输入：\n")


def prepare(rows: list[dict]) -> list[tuple[dict, str, str, core.Sample, set[str]]]:
    result = []
    for row in rows:
        sample = core.Sample(TRAIN / row["pack"], "train", row["sample_id"],
                             row["question_type"], row["question"])
        content, allowed, _ = workflow.training_view_input(sample, "A")
        payload = json.loads(content)
        if payload.pop("track") != "A":
            raise ValueError("Unexpected view")
        base = json.dumps(payload, ensure_ascii=False)
        if sha(base) != row["input_sha256"]:
            raise ValueError(f"Input changed: {sample.sample_id}")
        candidates = row["candidate_paths"]
        if len(candidates) > 5 or any(
                not isinstance(chain, list) or any(node not in allowed for node in chain)
                for chain in candidates):
            raise ValueError(f"Invalid frozen candidates: {sample.sample_id}")
        guided = GUIDANCE.format(candidates=json.dumps(candidates, ensure_ascii=False)) + base
        result.append((row, base, guided, sample, allowed))
    return result


def balance(key: str) -> Decimal:
    request = Request("https://api.deepseek.com/user/balance", headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        data = json.load(response)
    balances = {row["currency"]: row["total_balance"]
                for row in data["balance_infos"]}
    return Decimal(balances["CNY"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-questions", type=int, default=5)
    parser.add_argument("--warn-spend", type=Decimal, default=Decimal("5.00"))
    parser.add_argument("--stop-spend", type=Decimal, default=Decimal("8.00"))
    parser.add_argument("--min-balance", type=Decimal, default=Decimal("10.00"))
    args = parser.parse_args()
    if not (1 <= args.limit <= 50 and 1 <= args.workers <= 8
            and 1 <= args.batch_questions <= 10
            and 0 < args.warn_spend < args.stop_spend):
        raise ValueError("Invalid run limits")
    frozen = core.read_json(MANIFEST)
    rows = frozen["rows"]
    if len(rows) != 50 or frozen["v4_prompt_sha256"] != sha(PROMPT.read_bytes()):
        raise ValueError("Frozen manifest or V4 prompt changed")
    prepared = prepare(rows)
    system = PROMPT.read_text(encoding="utf-8")
    signature = {"manifest_sha256": sha(MANIFEST.read_bytes()),
                 "system_prompt_sha256": sha(system), "guidance_sha256": sha(GUIDANCE),
                 "model": MODEL, "api": "DeepSeek Chat Completions",
                 "response_format": {"type": "json_object"}, "max_tokens": 8192,
                 "temperature": "provider default", "reasoning_effort": "provider default",
                 "gold_sent": False, "view": "simulated A", "arms": ["base", "guided"]}
    if args.dry_run:
        print(json.dumps({"questions": 50, "calls": args.limit * 2,
                          "base_chars_total": sum(len(x[1]) for x in prepared[:args.limit]),
                          "guided_chars_total": sum(len(x[2]) for x in prepared[:args.limit]),
                          "base_max_chars": max(len(x[1]) for x in prepared[:args.limit]),
                          "signature": signature}, ensure_ascii=False))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "metadata.json"
    attempts_path = OUT / "attempts.jsonl"
    key = read_key()
    now = balance(key)
    if meta_path.exists():
        meta = core.read_json(meta_path)
        if any(meta.get(k) != v for k, v in signature.items()):
            raise ValueError("Resume metadata mismatch")
        origin = Decimal(meta["start_balance_cny"])
    else:
        origin = now
        core.atomic_json(meta_path, {**signature, "start_balance_cny": str(now),
                                     "stop_spend_cny": str(args.stop_spend),
                                     "min_balance_cny": str(args.min_balance)})
    if origin - now >= args.stop_spend or now <= args.min_balance:
        raise RuntimeError("Budget cap reached before calls")
    done = {}
    if attempts_path.exists():
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            done[(item["sample_id"], item["arm"])] = item
    old = historical_module()

    def call(item: tuple, arm: str) -> dict:
        row, base, guided, sample, allowed = item
        content = base if arm == "base" else guided
        attempt = call_api(key, MODEL, system, content, timeout=300, max_tokens=8192)
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
        return {"sample_id": row["sample_id"], "arm": arm,
                "input_sha256": sha(content), "parsed": parsed, "attempt": attempt}

    print(json.dumps({"started": True, "target_calls": args.limit * 2,
                      "remaining_calls": sum((row["sample_id"], arm) not in done
                                             for row in rows[:args.limit]
                                             for arm in ("base", "guided")),
                      "balance_cny": str(now)}, ensure_ascii=False), flush=True)
    for offset in range(0, args.limit, args.batch_questions):
        work = [(item, arm) for item in prepared[offset:offset + args.batch_questions]
                for arm in ("base", "guided")
                if (item[0]["sample_id"], arm) not in done]
        if not work:
            continue
        before = balance(key)
        if origin - before >= args.stop_spend or before <= args.min_balance:
            print(json.dumps({"budget_stop": True, "balance_cny": str(before)}), flush=True)
            break
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(call, item, arm) for item, arm in work]
            for future in as_completed(futures):
                record = future.result()
                with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[record["sample_id"], record["arm"]] = record
                print(json.dumps({"sample_id": record["sample_id"],
                                  "arm": record["arm"], "valid": record["parsed"] is not None,
                                  "elapsed": record["attempt"].get("elapsed_seconds"),
                                  "usage": record["attempt"].get("usage"),
                                  "error": record["attempt"].get("validation_error") or
                                           record["attempt"].get("error")},
                                 ensure_ascii=False), flush=True)
        after = balance(key)
        print(json.dumps({"batch_end": offset + len(work) // 2,
                          "balance_cny": str(after), "spent_cny": str(origin - after)}),
              flush=True)
        if origin - after >= args.warn_spend:
            print(json.dumps({"budget_warning": True,
                              "spent_cny": str(origin - after)}), flush=True)
        if origin - after >= args.stop_spend or after <= args.min_balance:
            break
    for arm in ("base", "guided"):
        predictions = [{"view": "A", **done[row["sample_id"], arm]["parsed"]}
                       for row in rows if (row["sample_id"], arm) in done
                       and done[row["sample_id"], arm]["parsed"] is not None]
        core.atomic_json(OUT / f"predictions_{arm}.json", predictions)
    print(json.dumps({"completed": len(done), "target": args.limit * 2,
                      "valid": sum(record["parsed"] is not None for record in done.values())},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
