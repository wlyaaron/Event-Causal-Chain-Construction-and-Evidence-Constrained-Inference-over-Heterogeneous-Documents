"""Simple local chain diagnostics on TRAIN gold only; not the official metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(predictions: Path, train_root: Path) -> dict:
    gold = {}
    for path in train_root.rglob("gold/问答对_答案.json"):
        for item in json.loads(path.read_text(encoding="utf-8")):
            gold[item["sample_id"]] = item
    records = json.loads(predictions.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("预测文件须为非空 JSON 数组")
    unknown = [r.get("sample_id") for r in records if r.get("sample_id") not in gold]
    if unknown:
        raise ValueError(f"预测中有 {len(unknown)} 个样本不属于训练集；不能读取测试集标准答案")
    exact = f1_total = refusals = correct_refusals = 0
    for record in records:
        expected = gold[record["sample_id"]]
        chains = expected.get("evidence_chains") or []
        actual = record["evidence_chain"]
        exact += int(actual in chains)
        best = 0.0
        for chain in chains:
            shared = len(set(actual) & set(chain))
            best = max(best, 2 * shared / (len(actual) + len(chain)) if actual or chain else 1.0)
        f1_total += best
        if not chains or all(not chain for chain in chains):
            refusals += 1
            correct_refusals += int(record["answer"] == "无法确定")
    n = len(records)
    return {"samples": n, "chain_exact_match": round(exact / n, 4),
            "best_chain_event_set_f1": round(f1_total / n, 4),
            "gold_empty_chain": refusals, "correct_refusals": correct_refusals,
            "note": "仅训练集证据链诊断；未计算答案事实、链边方向和置信度；不是官网分数"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions, args.train_root), ensure_ascii=False, indent=2))
