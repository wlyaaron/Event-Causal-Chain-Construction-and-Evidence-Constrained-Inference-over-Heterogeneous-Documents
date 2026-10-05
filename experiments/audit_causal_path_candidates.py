"""TRAIN-only diagnostic of old versus question-conditioned A path candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_baselines as metrics
import task4_core as core
import task4_workflow as workflow


def audit(selection_path: Path) -> dict:
    selection = core.read_json(selection_path)
    root = core.REPO_ROOT / "数据集" / "训练集"
    rows = []
    for spec in selection:
        pack = root / spec["pack"]
        question = next(item for item in core.read_json(pack / "问题.json")
                        if item["sample_id"] == spec["sample_id"])
        gold = next(item for item in core.read_json(pack / "gold" / "问答对_答案.json")
                    if item["sample_id"] == spec["sample_id"])
        sample = core.Sample(pack, "train", spec["sample_id"],
                             question["question_type"], question["question"])
        alternatives = [chain for chain in gold.get("evidence_chains") or []
                        if isinstance(chain, list) and chain]
        old = workflow.graph_path_hints(sample, limit=5)
        new = workflow.question_conditioned_paths(sample, limit=5)
        row = {"sample_id": sample.sample_id, "pack": pack.name,
               "question_type": sample.question_type,
               "gold_paths": alternatives}
        for name, paths in (("old", old), ("new", new)):
            row[name] = {"paths": paths,
                         "any_exact": int(any(path in alternatives for path in paths)),
                         "best_node_f1": max((metrics.set_f1(path, gold_chain)
                                              for path in paths for gold_chain in alternatives),
                                             default=None)}
        rows.append(row)
    report = {"samples": len(rows), "gold_nonempty": sum(bool(row["gold_paths"]) for row in rows),
              "old_any_exact": sum(row["old"]["any_exact"] for row in rows),
              "new_any_exact": sum(row["new"]["any_exact"] for row in rows),
              "note": "Candidate oracle coverage only; not a final answer or submission score.",
              "rows": rows}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
                        default=core.REPO_ROOT / "experiments" / "causal_path_holdout_20261005.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("candidate audit must stay under ignored outputs/")
    report = audit(args.selection)
    core.atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
