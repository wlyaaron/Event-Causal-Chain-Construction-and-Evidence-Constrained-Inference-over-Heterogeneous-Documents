"""Posthoc audit of safe format repair on saved V6 raw outputs.

The V6 outputs helped identify the format problem, so this audit is diagnostic,
not an independent estimate. The rule itself never reads gold.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
import task4_workflow as workflow  # noqa: E402
from task4_refusal_format import normalize_raw  # noqa: E402
from experiments.run_codex_initial_deepseek_protocol import historical_module  # noqa: E402


SOURCES = {
    "old90": ROOT / "experiments/reasoning_type_90_20261009.json",
    "fresh30_v6": ROOT / "experiments/reasoning_type_fresh_30_20261009.json",
}
ATTEMPTS = ROOT / "outputs/deepseek_refusal_four_way_v6_20261010"
SCORED = ROOT / "outputs/deepseek_refusal_four_way_v6_20261010/score/all_scores.json"
OUT = ROOT / "experiments/results/task4_refusal_format_existing_audit_20261010.json"


def main() -> None:
    gold_cases = core.read_json(SCORED)["sets"]
    old = historical_module()
    report = {"note": "Posthoc saved-output audit; V6 cases informed the repair rule.",
              "source_sha256": {}, "sets": {}}
    for name, manifest_path in SOURCES.items():
        path = ATTEMPTS / name / "attempts.jsonl"
        report["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = {row["sample_id"]: row for row in core.read_json(manifest_path)["rows"]}
        gold = {case["sample_id"]: case["gold_answers"]
                for case in gold_cases[name]["cases"]}
        changes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["arm"] != "v6":
                continue
            raw = item["attempt"].get("raw")
            if not isinstance(raw, str):
                continue
            repaired, tag = normalize_raw(raw)
            if tag is None:
                continue
            row = rows[item["sample_id"]]
            sample = core.Sample(ROOT / "数据集/训练集" / row["pack"], "train",
                                 row["sample_id"], row["question_type"], row["question"])
            _, allowed, _ = workflow.training_view_input(sample, row["view"])
            old_sample = old.Sample(sample.pack, row["view"], sample.sample_id,
                                    sample.question_type, sample.question)
            try:
                parsed = old.parse_prediction(repaired, old_sample, allowed)
            except (ValueError, TypeError, KeyError):
                parsed = None
            changes.append({"sample_id": sample.sample_id, "repair": tag,
                            "original_valid": item["parsed"] is not None,
                            "repaired_valid": parsed is not None,
                            "original_answer": json.loads(raw).get("answer"),
                            "repaired_answer": parsed["answer"] if parsed else None,
                            "gold_strict": gold[sample.sample_id] == "无法确定"})
        original = gold_cases[name]["summary"]["all"]["v6"]
        rescued = [case for case in changes
                   if not case["original_valid"] and case["repaired_valid"]]
        strict_rescued = sum(case["gold_strict"] for case in rescued)
        nonstrict_to_strict = sum(not case["gold_strict"] for case in rescued)
        report["sets"][name] = {
            "n": len(rows), "repairs": len(changes), "cases": changes,
            "valid_before": original["valid"],
            "valid_after": original["valid"] + len(rescued),
            "strict_gold_hit_before": original["strict_refusal_confusion"].get("tp", 0),
            "strict_gold_hit_after": original["strict_refusal_confusion"].get("tp", 0)
            + strict_rescued,
            "nonstrict_gold_strict_before": original["strict_refusal_confusion"].get("fp", 0),
            "nonstrict_gold_strict_after": original["strict_refusal_confusion"].get("fp", 0)
            + nonstrict_to_strict,
        }
    core.atomic_json(OUT, report)
    print(json.dumps({name: {"n": group["n"], "repairs": group["repairs"]}
                      for name, group in report["sets"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
