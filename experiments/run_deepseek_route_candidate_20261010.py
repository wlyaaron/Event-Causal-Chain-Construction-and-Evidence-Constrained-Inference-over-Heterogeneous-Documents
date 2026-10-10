"""Gold-blind, resumable DeepSeek route/candidate experiment on frozen fit30.

The old V4 predictions are the paired control. Model outputs remain research
artifacts under ignored outputs/ and must not be used for contest submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen

import task4_core as core
import task4_workflow as workflow
from experiments.run_deepseek_initial_protocol_validation import call_api, read_key
from experiments.task4_route_candidates import candidates, parse_route

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/reasoning_type_fresh_30_20261009.json"
BASELINE = ROOT / "outputs/deepseek_v4_20261009/fresh30"
OUT = ROOT / "outputs/deepseek_route_candidate_20261010"
ROUTE_PROMPT = ROOT / "experiments/prompts/task4_route_decision_20261010.txt"
FINAL_PROMPT = ROOT / "experiments/prompts/task4_route_finalize_20261010.txt"
MODEL = "deepseek-flash"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def prepare(rows: list[dict]) -> list[tuple[dict, str, core.Sample, set[str]]]:
    """Rebuild frozen public views without opening any gold files."""
    packs = {row["pack"] for row in rows}
    samples = {sample.sample_id: sample for sample in core.discover_samples(
        ROOT / "数据集/训练集", limit=0) if sample.pack.name in packs}
    prepared = []
    for row in rows:
        sample = samples[row["sample_id"]]
        if sample.pack.name != row["pack"]:
            raise ValueError("Frozen pack mismatch")
        content, allowed, _ = workflow.training_view_input(sample, row["view"])
        if sha(content) != row["input_sha256"]:
            raise ValueError(f"Frozen input changed: {row['sample_id']}")
        payload = json.loads(content)
        if payload.pop("track", None) != row["view"]:
            raise ValueError("Unexpected view field")
        prepared.append((row, json.dumps(payload, ensure_ascii=False), sample, allowed))
    return prepared


def balance(key: str) -> Decimal:
    request = Request("https://api.deepseek.com/user/balance", headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        data = json.load(response)
    return Decimal(next(row["total_balance"] for row in data["balance_infos"]
                        if row["currency"] == "CNY"))


def prior_predictions() -> dict[str, dict | None]:
    attempts = [json.loads(line) for line in (BASELINE / "attempts.jsonl").read_text(
        encoding="utf-8").splitlines()]
    chosen = {x["sample_id"]: x for x in attempts if x["arm"] == "v4"}
    return {sid: item["parsed"] for sid, item in chosen.items()}


def invoke(key: str, system: str, user: str, max_tokens: int) -> dict:
    result = call_api(key, MODEL, system, user, timeout=300,
                      retries=1, max_tokens=max_tokens)
    result["user_input_sha256"] = sha(user)
    return result


def process_one(item: tuple, previous: dict | None, key: str,
                route_prompt: str, final_prompt: str) -> dict:
    row, content, sample, allowed = item
    payload = json.loads(content)
    draft = {k: previous[k] for k in ("answer", "evidence_chain", "confidence")} if previous else None
    route_user = json.dumps({"original_input": payload, "v4_draft": draft},
                            ensure_ascii=False, separators=(",", ":"))
    first = invoke(key, route_prompt, route_user, 4096)
    route = None
    try:
        if first.get("raw") is None or first.get("finish_reason") == "length":
            raise ValueError(first.get("error") or "Missing or truncated route output")
        route = parse_route(first["raw"], allowed)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        first["validation_error"] = str(exc).replace(key, "[REDACTED]")
    record = {"sample_id": row["sample_id"], "view": row["view"],
              "route": route, "route_attempt": first, "candidates": [],
              "final_attempt": None, "prediction": None, "decision": "route_invalid"}
    if route is None:
        return record

    # The easy consensus refusal is standardized by code. A disagreement gets
    # a full-material second review, since erroneous refusal is costly.
    if route["route"] == "strict_refusal" and draft and draft["answer"] == "无法确定":
        record["prediction"] = {"sample_id": sample.sample_id,
                                "answer": "无法确定", "evidence_chain": [],
                                "confidence": None,
                                "question_type": None if sample.question_type == "unanswerable"
                                else sample.question_type}
        record["decision"] = "strict_consensus_code"
        return record

    choices = candidates(payload, draft, route)
    record["candidates"] = choices
    final_user = json.dumps({"original_input": payload, "v4_draft": draft,
                             "route_review": route, "candidate_chains": choices},
                            ensure_ascii=False, separators=(",", ":"))
    final = invoke(key, final_prompt, final_user, 8192)
    record["final_attempt"] = final
    try:
        if final.get("raw") is None or final.get("finish_reason") == "length":
            raise ValueError(final.get("error") or "Missing or truncated final output")
        record["prediction"] = core.parse_prediction(final["raw"], sample, allowed)
        record["decision"] = "model_review"
    except (ValueError, TypeError, KeyError) as exc:
        final["validation_error"] = str(exc).replace(key, "[REDACTED]")
        record["decision"] = "final_invalid"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--stop-spend-cny", type=Decimal, default=Decimal("3.00"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 30 or not 1 <= args.workers <= 3 or args.stop_spend_cny <= 0:
        raise ValueError("Invalid size, concurrency or budget")
    frozen = core.read_json(MANIFEST)
    if len(frozen["rows"]) != 30:
        raise ValueError("Frozen manifest changed")
    rows = frozen["rows"][:args.limit]
    prepared = prepare(rows)
    previous = prior_predictions()
    if set(previous) != {r["sample_id"] for r in frozen["rows"]}:
        raise ValueError("V4 baseline is incomplete")
    baseline_meta = core.read_json(BASELINE / "metadata.json")
    if baseline_meta["manifest_sha256"] != sha(MANIFEST.read_bytes()):
        raise ValueError("V4 baseline uses a different manifest")
    route_prompt, final_prompt = (p.read_text(encoding="utf-8")
                                  for p in (ROUTE_PROMPT, FINAL_PROMPT))
    signature = {"manifest_sha256": sha(MANIFEST.read_bytes()),
                 "baseline_attempts_sha256": sha((BASELINE / "attempts.jsonl").read_bytes()),
                 "route_prompt_sha256": sha(route_prompt), "final_prompt_sha256": sha(final_prompt),
                 "model": MODEL, "gold_sent": False, "max_route_tokens": 4096,
                 "max_final_tokens": 8192, "response_format": "json_object",
                 "stage1_requests_max": 30, "stage2_requests_max": 30,
                 "stage2_skip": "strict route and strict V4 consensus"}
    if args.dry_run:
        print(json.dumps({"signature": signature, "selected": len(rows),
                          "max_new_requests": 2 * len(rows),
                          "input_chars_sum": sum(len(p[1]) for p in prepared),
                          "max_input_chars": max(len(p[1]) for p in prepared)},
                         ensure_ascii=False))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    meta_path, attempts_path = OUT / "metadata.json", OUT / "attempts.jsonl"
    key = read_key()
    now = balance(key)
    if meta_path.exists():
        meta = core.read_json(meta_path)
        if any(meta.get(k) != v for k, v in signature.items()):
            raise ValueError("Resume metadata mismatch")
    else:
        core.atomic_json(meta_path, {**signature, "start_balance_cny": str(now),
                                     "stop_spend_cny": str(args.stop_spend_cny)})
        meta = core.read_json(meta_path)
    origin = Decimal(meta["start_balance_cny"])
    cap = min(args.stop_spend_cny, Decimal(meta["stop_spend_cny"]))
    done = {}
    if attempts_path.exists():
        done = {x["sample_id"]: x for x in (json.loads(line) for line in
                attempts_path.read_text(encoding="utf-8").splitlines())}
    print(json.dumps({"selected": len(rows), "done": len(done),
                      "balance_cny": str(now), "spend_cap_cny": str(cap)}), flush=True)
    for offset in range(0, len(prepared), args.workers):
        pending = [item for item in prepared[offset:offset + args.workers]
                   if item[0]["sample_id"] not in done]
        if not pending:
            continue
        before = balance(key)
        if origin - before >= cap or before <= Decimal("1.00"):
            print(json.dumps({"budget_stop": True, "balance_cny": str(before)}), flush=True)
            break
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, item, previous[item[0]["sample_id"]],
                                   key, route_prompt, final_prompt): item[0]["sample_id"]
                       for item in pending}
            for future in as_completed(futures):
                result = future.result()
                with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                done[result["sample_id"]] = result
                print(json.dumps({"sample_id": result["sample_id"],
                                  "route": (result["route"] or {}).get("route"),
                                  "decision": result["decision"]}, ensure_ascii=False), flush=True)
        after = balance(key)
        print(json.dumps({"completed": len(done), "balance_cny": str(after),
                          "observed_drop_cny": str(origin-after)}), flush=True)
    predictions = [{"view": row["view"], **done[row["sample_id"]]["prediction"]}
                   for row in frozen["rows"] if row["sample_id"] in done
                   and done[row["sample_id"]]["prediction"] is not None]
    core.atomic_json(OUT / "predictions.json", predictions)
    print(json.dumps({"completed": len(done), "target": args.limit,
                      "valid_predictions": len(predictions),
                      "balance_cny": str(balance(key))}), flush=True)


if __name__ == "__main__":
    main()
