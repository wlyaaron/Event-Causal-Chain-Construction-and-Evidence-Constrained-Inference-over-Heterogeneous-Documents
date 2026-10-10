"""Score raw versus conservatively normalized outputs on the frozen 24."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
import task4_workflow as workflow  # noqa: E402
from experiments import score_deepseek_v4_staged_20261009 as scorer  # noqa: E402
from task4_refusal_format import normalize_raw  # noqa: E402
from experiments.run_codex_initial_deepseek_protocol import historical_module  # noqa: E402


MANIFEST = ROOT / "experiments/refusal_format24_20261010.json"
RAW_OUT = ROOT / "outputs/deepseek_refusal_format24_20261010"
NORMALIZED_OUT = RAW_OUT / "normalized"
PUBLIC_OUT = ROOT / "experiments/results/task4_refusal_format24_20261010.json"
ARMS = ("v4", "fmt")


def configure(out: Path) -> None:
    scorer.OUT = out
    scorer.SCORE = out / "score"
    scorer.MANIFESTS = {"format24": MANIFEST}
    scorer.ARMS = {"format24": ARMS}


def normalize_attempts() -> dict[str, dict[str, str | None]]:
    rows = core.read_json(MANIFEST)["rows"]
    by_id = {row["sample_id"]: row for row in rows}
    old = historical_module()
    source = RAW_OUT / "format24/attempts.jsonl"
    attempts = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if {(item["sample_id"], item["arm"]) for item in attempts} != {
        (row["sample_id"], arm) for row in rows for arm in ARMS
    }:
        raise ValueError("Incomplete or duplicate paired attempts")
    tags: dict[str, dict[str, str | None]] = {arm: {} for arm in ARMS}
    normalized = []
    for item in attempts:
        row = by_id[item["sample_id"]]
        arm = item["arm"]
        raw = item["attempt"].get("raw")
        repaired, tag = normalize_raw(raw) if isinstance(raw, str) else (raw, None)
        tags[arm][item["sample_id"]] = tag
        copy = {**item, "attempt": dict(item["attempt"]), "format_repair": tag}
        if tag is not None:
            sample = core.Sample(ROOT / "数据集/训练集" / row["pack"], "train",
                                 row["sample_id"], row["question_type"], row["question"])
            _, allowed, _ = workflow.training_view_input(sample, row["view"])
            old_sample = old.Sample(sample.pack, row["view"], sample.sample_id,
                                    sample.question_type, sample.question)
            try:
                copy["parsed"] = old.parse_prediction(repaired, old_sample, allowed)
            except (ValueError, TypeError, KeyError) as exc:
                copy["parsed"] = None
                copy["attempt"]["validation_error"] = str(exc)
            else:
                copy["attempt"]["original_validation_error"] = copy["attempt"].pop(
                    "validation_error", None)
                copy["normalized_raw"] = repaired
        normalized.append(copy)
    target = NORMALIZED_OUT / "format24"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "attempts.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in normalized:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    for arm in ARMS:
        predictions = [{"view": by_id[item["sample_id"]]["view"], **item["parsed"]}
                       for item in normalized if item["arm"] == arm
                       and item["parsed"] is not None]
        core.atomic_json(target / f"predictions_{arm}.json", predictions)
    return tags


def main() -> None:
    configure(RAW_OUT)
    scorer.main()
    raw = core.read_json(RAW_OUT / "score/all_scores.json")["sets"]["format24"]
    tags = normalize_attempts()
    configure(NORMALIZED_OUT)
    scorer.main()
    normalized = core.read_json(NORMALIZED_OUT / "score/all_scores.json")["sets"]["format24"]
    original = {row["sample_id"]: row for row in raw["cases"]}
    final = {row["sample_id"]: row for row in normalized["cases"]}
    rows = []
    for sid, row in original.items():
        counterpart = final[sid]
        rows.append({"sample_id": sid, "view": row["view"],
                     "question_type": row["question_type"], "question": row["question"],
                     "gold_answer": row["gold_answers"],
                     "gold_chains": row["gold_chains"],
                     "arms": {arm: {"raw": row["results"][arm],
                                    "normalized": counterpart["results"][arm],
                                    "format_repair": tags[arm][sid]}
                              for arm in ARMS}})
    report = {"manifest": str(MANIFEST), "n": len(rows),
              "note": "Fit development subset; no gold was sent to API. Raw and normalized outputs use the same model attempts; invalid rows stay in the denominator.",
              "raw_summary": raw["summary"],
              "normalized_summary": normalized["summary"], "rows": rows}
    PUBLIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    core.atomic_json(PUBLIC_OUT, report)
    compact = {kind: {arm: summary["all"][arm] for arm in ARMS}
               for kind, summary in (("raw", raw["summary"]),
                                     ("normalized", normalized["summary"]))}
    print(json.dumps({"summary": compact, "format_repairs": {
        arm: sum(tag is not None for tag in tags[arm].values()) for arm in ARMS},
        "output": str(PUBLIC_OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
