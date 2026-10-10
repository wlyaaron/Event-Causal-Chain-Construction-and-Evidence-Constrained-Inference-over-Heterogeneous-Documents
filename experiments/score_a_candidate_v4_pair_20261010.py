"""Score the completed 50-question paired run against intact organizer train gold."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_baselines as basic  # noqa: E402
import task4_core as core  # noqa: E402
from experiments.evaluate_validation_diagnostic import evaluate, gold_map  # noqa: E402
from experiments.freeze_a_candidate_v4_pair_20261010 import OUT as MANIFEST, TRAIN  # noqa: E402
from experiments.run_a_candidate_v4_pair_20261010 import OUT  # noqa: E402

SCORE = OUT / "score"
EMBEDDING = (ROOT / "outputs" / "hf_cache" / "hub" /
             "models--BAAI--bge-small-zh-v1.5" / "snapshots" /
             "7999e1d3359715c523056ef9478215996d62a620")
METRICS = ("answer_char_f1", "answer_semantic_cosine", "required_fact_coverage_lit",
           "chain_node_f1", "chain_edge_f1", "chain_exact", "strict_refusal_correct",
           "answer_right_chain_wrong")


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(cases: list[dict], arm: str) -> dict:
    results = [case["results"][arm] for case in cases]
    summary = {"n": len(cases), "valid": sum(result["valid"] for result in results),
               "invalid": sum(not result["valid"] for result in results),
               "chain_exact_count": sum(result["metrics"]["chain_exact"] or 0
                                        for result in results),
               "numeric_confidence_mean": mean([
                   result["prediction"]["confidence"] for result in results
                   if result["prediction"] and
                   result["prediction"]["confidence"] is not None]),
               "answer_chars_valid_mean": mean([
                   len(result["prediction"]["answer"]) for result in results
                   if result["prediction"]]),
               "prompt_tokens": sum((result["usage"] or {}).get("prompt_tokens", 0)
                                    for result in results),
               "completion_tokens": sum((result["usage"] or {}).get("completion_tokens", 0)
                                        for result in results),
               "latency_seconds_sum": round(sum(result["elapsed_seconds"] or 0
                                                for result in results), 2)}
    for metric in METRICS:
        summary[metric] = mean([float(result["metrics"][metric]) for result in results
                                if result["metrics"][metric] is not None])
    confusion = Counter()
    for case in cases:
        gold_strict = case["gold_answers"] == "无法确定"
        prediction = case["results"][arm]["prediction"]
        said_strict = bool(prediction and prediction["answer"] == "无法确定")
        confusion["tp" if gold_strict and said_strict else
                  "fn" if gold_strict else "fp" if said_strict else "tn"] += 1
    summary["strict_refusal_confusion"] = dict(confusion)
    return summary


def score_arm(rows: list[dict], attempts: dict, gold: dict, arm: str) -> dict:
    expected = {row["sample_id"] for row in rows}
    selected = {sid: item for (sid, name), item in attempts.items() if name == arm}
    if set(selected) != expected:
        raise ValueError(f"Incomplete arm {arm}: {len(selected)}/{len(expected)}")
    predictions = core.read_json(OUT / f"predictions_{arm}.json")
    valid_ids = {prediction["sample_id"] for prediction in predictions}
    if valid_ids != {sid for sid, item in selected.items() if item["parsed"] is not None}:
        raise ValueError(f"Prediction/attempt mismatch: {arm}")
    metrics_by_id = {}
    if predictions:
        valid_manifest = SCORE / f"valid_manifest_{arm}.json"
        core.atomic_json(valid_manifest, {"rows": [{**row, "risk_flags": []}
                                                   for row in rows if row["sample_id"] in valid_ids]})
        evaluation = evaluate(valid_manifest, {arm: OUT / f"predictions_{arm}.json"},
                              TRAIN, str(EMBEDDING))
        core.atomic_json(SCORE / f"gold_metrics_{arm}.json", evaluation)
        metrics_by_id = {row["sample_id"]: row["metrics"][arm]
                         for row in evaluation["rows"]}
    scored = {}
    for row in rows:
        sid = row["sample_id"]
        item = selected[sid]
        facts_available = bool(basic.facts(gold[sid], "required_facts"))
        metrics = metrics_by_id.get(sid)
        if metrics is None:
            metrics = {key: (None if key == "required_fact_coverage_lit"
                             and not facts_available else 0) for key in METRICS}
        scored[sid] = {"valid": item["parsed"] is not None,
                       "metrics": {key: metrics.get(key) for key in METRICS},
                       "prediction": item["parsed"],
                       "raw": item["attempt"].get("raw"),
                       "error": item["attempt"].get("validation_error") or
                                item["attempt"].get("error"),
                       "usage": item["attempt"].get("usage"),
                       "response_model": item["attempt"].get("response_model"),
                       "elapsed_seconds": item["attempt"].get("elapsed_seconds")}
    return scored


def paired_delta(cases: list[dict], metric: str) -> dict:
    differences = [float(case["results"]["guided"]["metrics"][metric])
                   - float(case["results"]["base"]["metrics"][metric])
                   for case in cases if all(case["results"][arm]["metrics"][metric]
                                            is not None for arm in ("base", "guided"))]
    if not differences:
        return {"n": 0, "mean_delta": None, "ci95": None}
    rng = random.Random(20261010)
    boot = sorted(sum(rng.choice(differences) for _ in differences) / len(differences)
                  for _ in range(10000))
    return {"n": len(differences),
            "mean_delta": round(sum(differences) / len(differences), 4),
            "ci95": [round(boot[250], 4), round(boot[9750], 4)]}


def main() -> None:
    SCORE.mkdir(parents=True, exist_ok=True)
    rows = core.read_json(MANIFEST)["rows"]
    lines = (OUT / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    if len(lines) != 100:
        raise ValueError(f"Expected 100 requests, found {len(lines)}")
    attempts = {}
    for line in lines:
        item = json.loads(line)
        key = item["sample_id"], item["arm"]
        if key in attempts:
            raise ValueError(f"Duplicate request: {key}")
        attempts[key] = item
    gold = gold_map(TRAIN, {row["pack"] for row in rows})
    results = {arm: score_arm(rows, attempts, gold, arm)
               for arm in ("base", "guided")}
    cases = []
    for row in rows:
        sid = row["sample_id"]
        gold_chains = gold[sid].get("evidence_chains") or []
        intact = {tuple(chain) for chain in gold_chains}
        case = {**row, "gold_answers": gold[sid].get("answers"),
                "gold_chains": gold_chains,
                "candidate_hit": bool(intact & {tuple(path) for path in row["candidate_paths"]}),
                "results": {arm: results[arm][sid] for arm in ("base", "guided")}}
        cases.append(case)
    categories = sorted(set(case["category"] for case in cases))
    summary = {"all": {arm: summarize(cases, arm) for arm in ("base", "guided")}}
    for category in categories:
        subset = [case for case in cases if case["category"] == category]
        summary[category] = {arm: summarize(subset, arm) for arm in ("base", "guided")}
        summary[category]["candidate_hit"] = sum(case["candidate_hit"] for case in subset)
    summary["all"]["candidate_hit"] = sum(case["candidate_hit"] for case in cases)
    paired = {metric: paired_delta(cases, metric) for metric in METRICS
              if metric != "answer_right_chain_wrong"}
    changes = Counter()
    for case in cases:
        b = bool(case["results"]["base"]["metrics"]["chain_exact"])
        g = bool(case["results"]["guided"]["metrics"]["chain_exact"])
        changes["gain" if g and not b else "loss" if b and not g else
                "both_hit" if b else "both_miss"] += 1
    report = {"scope": "holdout packages of organizer training data, A view",
              "note": "Original complete gold alternatives; local diagnostic metrics, not official test score. Invalid model outputs receive zero; no gold was sent to the model.",
              "manifest": str(MANIFEST), "n": 50, "summary": summary,
              "paired_guided_minus_base": paired,
              "chain_exact_pairing": dict(changes), "cases": cases}
    core.atomic_json(SCORE / "all_scores.json", report)
    print(json.dumps({"summary": summary, "paired": paired,
                      "chain_exact_pairing": dict(changes),
                      "output": str(SCORE / "all_scores.json")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
