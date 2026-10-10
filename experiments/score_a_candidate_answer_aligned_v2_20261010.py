"""Score the completed exploratory rerun against intact organizer gold."""
from __future__ import annotations

import json
from pathlib import Path

import evaluate_baselines as basic
import task4_core as core
from experiments.evaluate_validation_diagnostic import evaluate, gold_map
from experiments.freeze_a_candidate_v4_pair_20261010 import OUT as MANIFEST, TRAIN
from experiments.run_a_candidate_answer_aligned_v2_20261010 import OUT
from experiments.score_a_candidate_v4_pair_20261010 import EMBEDDING, METRICS, summarize


SCORE = OUT / "score"


def main() -> None:
    rows = core.read_json(MANIFEST)["rows"]
    attempts = {}
    for line in (OUT / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        sid = item["sample_id"]
        if sid in attempts:
            raise ValueError(f"Duplicate {sid}")
        attempts[sid] = item
    if set(attempts) != {row["sample_id"] for row in rows}:
        raise ValueError("Run incomplete")
    predictions = core.read_json(OUT / "predictions.json")
    valid_ids = {x["sample_id"] for x in predictions}
    if valid_ids != {sid for sid, a in attempts.items() if a["parsed"] is not None}:
        raise ValueError("Prediction mismatch")
    SCORE.mkdir(parents=True, exist_ok=True)
    gold = gold_map(TRAIN, {row["pack"] for row in rows})
    metric_by_id = {}
    if predictions:
        manifest = SCORE / "valid_manifest.json"
        core.atomic_json(manifest, {"rows": [{**row, "risk_flags": []}
                                             for row in rows if row["sample_id"] in valid_ids]})
        evaluation = evaluate(manifest, {"new": OUT / "predictions.json"},
                              TRAIN, str(EMBEDDING))
        core.atomic_json(SCORE / "gold_metrics.json", evaluation)
        metric_by_id = {x["sample_id"]: x["metrics"]["new"]
                        for x in evaluation["rows"]}
    cases = []
    for row in rows:
        sid = row["sample_id"]
        item = attempts[sid]
        facts_available = bool(basic.facts(gold[sid], "required_facts"))
        metrics = metric_by_id.get(sid)
        if metrics is None:
            metrics = {key: (None if key == "required_fact_coverage_lit"
                             and not facts_available else 0) for key in METRICS}
        result = {"valid": item["parsed"] is not None,
                  "metrics": {key: metrics.get(key) for key in METRICS},
                  "prediction": item["parsed"],
                  "error": item["attempt"].get("validation_error") or item["attempt"].get("error"),
                  "usage": item["attempt"].get("usage"),
                  "response_model": item["attempt"].get("response_model"),
                  "elapsed_seconds": item["attempt"].get("elapsed_seconds")}
        cases.append({**row, "gold_answers": gold[sid].get("answers"),
                      "gold_chains": gold[sid].get("evidence_chains") or [],
                      "results": {"new": result}})
    summary = summarize(cases, "new")
    report = {"scope": "exploratory repeated holdout-50", "gold_sent": False,
              "warning": "The same 50 cases informed the prompt; not independent efficacy validation.",
              "summary": summary, "cases": cases}
    core.atomic_json(SCORE / "all_scores.json", report)
    print(json.dumps({"summary": summary, "output": str(SCORE / "all_scores.json")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
