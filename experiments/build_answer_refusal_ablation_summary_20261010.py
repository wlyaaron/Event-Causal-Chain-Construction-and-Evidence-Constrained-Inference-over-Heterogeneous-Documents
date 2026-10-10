"""Export compact, inspectable paired evidence for answer/refusal experiments.

Only saved scored results are read. The original API attempt logs remain local.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/task4_answer_refusal_ablation_20261010.json"
SOURCES = {
    "candidate_original": ROOT / "outputs/a_candidate_v4_pair_holdout50_20261010/score/all_scores.json",
    "candidate_v2": ROOT / "outputs/a_candidate_answer_aligned_v2_20261010/score/all_scores.json",
    "candidate_chain_only": ROOT / "outputs/a_candidate_chain_only_20261010/score/all_scores.json",
    "v4": ROOT / "outputs/deepseek_v4_20261009/score/all_scores.json",
    "refusal_v5": ROOT / "outputs/deepseek_refusal_four_way_v5_20261010/score/all_scores.json",
    "refusal_v6": ROOT / "outputs/deepseek_refusal_four_way_v6_20261010/score/all_scores.json",
}


def compact_result(result: dict) -> dict:
    return {"valid": result["valid"], "prediction": result["prediction"],
            "metrics": result["metrics"], "error": result.get("error")}


def indexed(cases: list[dict]) -> dict[str, dict]:
    result = {case["sample_id"]: case for case in cases}
    if len(result) != len(cases):
        raise ValueError("Duplicate sample ID")
    return result


def main() -> None:
    data = {name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in SOURCES.items()}
    original = indexed(data["candidate_original"]["cases"])
    v2 = indexed(data["candidate_v2"]["cases"])
    chain_only = indexed(data["candidate_chain_only"]["cases"])
    if not (len(original) == 50 and set(original) == set(v2) == set(chain_only)):
        raise ValueError("Answer experiment sets differ")
    answer_rows = []
    for sid, case in original.items():
        answer_rows.append({"sample_id": sid, "question": case["question"],
                            "question_type": case["question_type"],
                            "category": case["category"],
                            "candidate_paths": case["candidate_paths"],
                            "gold_answer": case["gold_answers"],
                            "gold_chains": case["gold_chains"],
                            "base": compact_result(case["results"]["base"]),
                            "original_candidate": compact_result(case["results"]["guided"]),
                            "v2": compact_result(v2[sid]["results"]["new"]),
                            "chain_only": compact_result(chain_only[sid]["results"]["new"])})
    refusal_sets = {}
    for v6_name, v4_name in (("old90", "old90"), ("fresh30_v6", "fresh30")):
        old = indexed(data["v4"]["sets"][v4_name]["cases"])
        new = indexed(data["refusal_v6"]["sets"][v6_name]["cases"])
        if set(old) != set(new):
            raise ValueError(f"Refusal experiment sets differ: {v6_name}")
        v5 = (indexed(data["refusal_v5"]["sets"]["old90"]["cases"])
              if v6_name == "old90" else {})
        refusal_sets[v6_name] = [{
            "sample_id": sid, "question": case["question"],
            "question_type": case["question_type"],
            "reasoning_type": case["reasoning_type"],
            "gold_answer": case["gold_answers"],
            "gold_chains": case["gold_chains"],
            "v4": compact_result(case["results"]["v4"]),
            "v5": compact_result(v5[sid]["results"]["v5"]) if v5 else None,
            "v6": compact_result(new[sid]["results"]["v6"]),
        } for sid, case in old.items()]
    payload = {
        "note": "Research only; original full gold alternatives; old90 and 50 were used for prompt development.",
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in SOURCES.items()},
        "answer50": {"summary": {
            "base": data["candidate_original"]["summary"]["all"]["base"],
            "original_candidate": data["candidate_original"]["summary"]["all"]["guided"],
            "v2": data["candidate_v2"]["summary"],
            "chain_only": data["candidate_chain_only"]["summary"],
        }, "rows": answer_rows},
        "refusal": {name: {"v4_summary": data["v4"]["sets"][v4_name]["summary"]["all"]["v4"],
                            "v5_summary": data["refusal_v5"]["sets"]["old90"]["summary"]["all"]["v5"]
                            if name == "old90" else None,
                            "v6_summary": data["refusal_v6"]["sets"][name]["summary"]["all"]["v6"],
                            "rows": refusal_sets[name]}
                    for name, v4_name in (("old90", "old90"), ("fresh30_v6", "fresh30"))},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"answer_rows": len(answer_rows),
                      "refusal_rows": {key: len(value["rows"])
                                       for key, value in payload["refusal"].items()},
                      "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
