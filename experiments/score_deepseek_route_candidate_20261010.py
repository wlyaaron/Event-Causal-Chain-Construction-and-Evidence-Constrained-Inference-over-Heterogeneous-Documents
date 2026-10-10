"""Gold-side candidate audit and paired scoring for the fit30 research run."""
from __future__ import annotations

import argparse
import json
from collections import Counter

import evaluate_baselines as basic
import task4_core as core
from experiments.evaluate_validation_diagnostic import evaluate, gold_map
from experiments.run_deepseek_route_candidate_20261010 import ROOT, prepare
from experiments.task4_route_candidates import candidates

MANIFEST = ROOT / "experiments/reasoning_type_fresh_30_20261009.json"
BASELINE = ROOT / "outputs/deepseek_v4_20261009/fresh30/predictions_v4.json"
OUT = ROOT / "outputs/deepseek_route_candidate_20261010"
SCORE = OUT / "score"
EMBEDDING = (ROOT / "outputs/hf_cache/hub/models--BAAI--bge-small-zh-v1.5/"
             "snapshots/7999e1d3359715c523056ef9478215996d62a620")
FIELDS = ("answer_char_f1", "answer_semantic_cosine", "required_fact_coverage_lit",
          "chain_node_f1", "chain_edge_f1", "chain_exact", "strict_refusal_correct")


def average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def audit_candidates(rows: list[dict], gold: dict[str, dict]) -> dict:
    """Check oracle coverage only; use original complete gold alternatives."""
    baseline = {x["sample_id"]: x for x in core.read_json(BASELINE)}
    results = []
    for row, content, _, _ in prepare(rows):
        sid = row["sample_id"]
        alternatives = gold[sid].get("evidence_chains") or []
        strict = gold[sid].get("answers") == "无法确定"
        draft = baseline.get(sid)
        if strict:
            continue
        pools = {}
        for mode in ("anchor", "path"):
            route = {"route": "answer", "evidence_mode": mode, "basis_ids": []}
            pool = candidates(json.loads(content), draft, route)
            pools[mode] = {"count": len(pool),
                           "recall": any(item["chain"] in alternatives for item in pool),
                           "chains": pool}
        results.append({"sample_id": sid, "view": row["view"],
                        "baseline_exact": bool(draft and draft["evidence_chain"] in alternatives),
                        "anchor": pools["anchor"], "path": pools["path"]})
    return {"n_non_strict": len(results),
            "baseline_exact": sum(x["baseline_exact"] for x in results),
            "anchor_oracle_recall": sum(x["anchor"]["recall"] for x in results),
            "path_oracle_recall": sum(x["path"]["recall"] for x in results),
            "rows": results,
            "note": "Upper-bound candidate recall, not model quality; original full alternatives only."}


def score_arm(rows: list[dict], name: str, path) -> dict[str, dict]:
    predictions = core.read_json(path)
    valid_ids = {x["sample_id"] for x in predictions}
    if len(valid_ids) != len(predictions):
        raise ValueError(f"Duplicate prediction in {name}")
    manifest = SCORE / f"valid_{name}.json"
    core.atomic_json(manifest, {"rows": [{**row, "risk_flags": []} for row in rows
                                         if row["sample_id"] in valid_ids]})
    report = evaluate(manifest, {name: path}, ROOT / "数据集/训练集", str(EMBEDDING))
    core.atomic_json(SCORE / f"gold_metrics_{name}.json", report)
    return {item["sample_id"]: item["metrics"][name] for item in report["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", action="store_true")
    args = parser.parse_args()
    rows = core.read_json(MANIFEST)["rows"]
    gold = gold_map(ROOT / "数据集/训练集", {r["pack"] for r in rows})
    SCORE.mkdir(parents=True, exist_ok=True)
    if args.candidate_audit:
        result = audit_candidates(rows, gold)
        core.atomic_json(SCORE / "candidate_oracle_audit.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                         ensure_ascii=False))
        return

    attempts = [json.loads(line) for line in (OUT / "attempts.jsonl").read_text(
        encoding="utf-8").splitlines()]
    by_id = {x["sample_id"]: x for x in attempts}
    if len(attempts) != 30 or set(by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("Run incomplete; do not score as full 30")
    baseline = {x["sample_id"]: x for x in core.read_json(BASELINE)}
    method = {x["sample_id"]: x for x in core.read_json(OUT / "predictions.json")}
    scores = {"v4": score_arm(rows, "v4", BASELINE),
              "method": score_arm(rows, "method", OUT / "predictions.json")}
    cases = []
    for row in rows:
        sid = row["sample_id"]
        answer = gold[sid].get("answers")
        strict = answer == "无法确定"
        result = by_id[sid]
        metrics = {arm: {field: scores[arm].get(sid, {}).get(field, 0)
                         for field in FIELDS} for arm in scores}
        cases.append({"sample_id": sid, "view": row["view"],
                      "reasoning_type": row["reasoning_type"],
                      "question_type": row["question_type"], "question": row["question"],
                      "gold_answer": answer, "gold_chains": gold[sid].get("evidence_chains"),
                      "gold_strict": strict, "v4": baseline.get(sid),
                      "method": method.get(sid), "route": result["route"],
                      "decision": result["decision"], "candidates": result["candidates"],
                      "metrics": metrics})

    def summarize(subset: list[dict], arm: str) -> dict:
        predictions = [x[arm] for x in subset]
        confusion = Counter()
        for case in subset:
            pred = case[arm]
            said_strict = bool(pred and pred["answer"] == "无法确定")
            confusion["tp" if case["gold_strict"] and said_strict else
                      "fn" if case["gold_strict"] else
                      "fp" if said_strict else "tn"] += 1
        return {"n": len(subset), "valid": sum(x is not None for x in predictions),
                "strict_confusion": dict(confusion),
                **{field: average([float(x["metrics"][arm][field]) for x in subset
                                   if x["metrics"][arm][field] is not None])
                   for field in FIELDS}}

    groups = {"all": cases,
              "strict_gold": [x for x in cases if x["gold_strict"]],
              "non_strict_gold": [x for x in cases if not x["gold_strict"]],
              **{view: [x for x in cases if x["view"] == view] for view in "ABC"}}
    summary = {name: {arm: summarize(group, arm) for arm in scores}
               for name, group in groups.items() if group}
    usage = Counter()
    latency = 0.0
    calls = 0
    for item in attempts:
        for stage in ("route_attempt", "final_attempt"):
            attempt = item.get(stage)
            if attempt is None:
                continue
            calls += 1
            latency += attempt.get("elapsed_seconds") or 0
            for key in ("prompt_tokens", "completion_tokens"):
                usage[key] += (attempt.get("usage") or {}).get(key, 0)
    report = {"n": len(cases), "summary": summary, "calls": calls,
              "usage": dict(usage), "latency_seconds_sum": round(latency, 2),
              "cases": cases,
              "note": "Exploratory fit30; original complete gold proxy, not official scoring."}
    core.atomic_json(SCORE / "all_scores.json", report)
    print(json.dumps({"summary": summary, "calls": calls, "usage": dict(usage),
                      "latency_seconds_sum": round(latency, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
