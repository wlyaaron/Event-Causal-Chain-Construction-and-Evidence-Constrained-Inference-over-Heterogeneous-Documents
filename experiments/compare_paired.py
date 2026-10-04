"""Paired, gold-backed TRAIN-only comparison of two prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_baselines as metrics
import task4_core as core


def score(record: dict, item: dict) -> dict[str, float]:
    chains = item.get("evidence_chains") or [[]]
    answers = metrics.answer_references(item)
    actual = record["evidence_chain"]
    edges = lambda chain: [f"{a}>{b}" for a, b in zip(chain, chain[1:])]
    return {
        "answer_char_f1": max((metrics.char_f1(record["answer"], value)
                               for value in answers), default=0.0),
        "chain_exact": float(actual in chains),
        "chain_node_f1": max(metrics.set_f1(actual, chain) for chain in chains),
        "chain_edge_f1": max(metrics.set_f1(edges(actual), edges(chain))
                             for chain in chains),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    args = parser.parse_args()
    baseline = {row["sample_id"]: row for row in core.read_json(args.baseline)}
    candidate = {row["sample_id"]: row for row in core.read_json(args.candidate)}
    gold = {}
    for path in args.train_root.rglob("gold/问答对_答案.json"):
        for item in core.read_json(path):
            gold[item["sample_id"]] = item
    rows = []
    for sample_id, prediction in candidate.items():
        if sample_id not in baseline or sample_id not in gold:
            raise ValueError(f"No paired baseline/gold: {sample_id}")
        old = score(baseline[sample_id], gold[sample_id])
        new = score(prediction, gold[sample_id])
        rows.append({"sample_id": sample_id,
                     "question_type": gold[sample_id].get("question_type"),
                     "baseline": old, "candidate": new,
                     "delta": {key: round(new[key] - old[key], 4) for key in old}})
    keys = ("answer_char_f1", "chain_exact", "chain_node_f1", "chain_edge_f1")
    summary = {key: {"baseline": round(sum(row["baseline"][key] for row in rows) / len(rows), 4),
                     "candidate": round(sum(row["candidate"][key] for row in rows) / len(rows), 4),
                     "delta": round(sum(row["delta"][key] for row in rows) / len(rows), 4)}
               for key in keys}
    print(json.dumps({"samples": len(rows), "summary": summary, "rows": rows},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
