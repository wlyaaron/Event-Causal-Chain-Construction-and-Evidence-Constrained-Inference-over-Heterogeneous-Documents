"""Show public score weights applied to explicitly named local TRAIN proxies.

Input is the JSON report from evaluate_baselines.py with --train-root and an
embedding model. This is a sensitivity ledger, not an official score estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def breakdown(group: dict) -> dict:
    required = ("answer_char_f1", "answer_semantic_cosine", "chain_event_f1",
                "chain_edge_f1", "chain_exact", "required_fact_coverage")
    missing = [key for key in required if key not in group]
    if missing:
        raise ValueError(f"诊断结果缺少指标 {missing}；语义相似度需要 --embedding-model")
    parts = {
        "facts_25_literal_proxy": 25 * group["required_fact_coverage"],
        "answer_char_10_proxy": 10 * group["answer_char_f1"],
        "answer_semantic_5_cosine_proxy": 5 * max(0, min(1, group["answer_semantic_cosine"])),
        "evidence_nodes_25_f1_proxy": 25 * group["chain_event_f1"],
        "structure_15_half_edge_f1_half_exact_proxy":
            15 * (group["chain_edge_f1"] + group["chain_exact"]) / 2,
    }
    result = {key: round(value, 3) for key, value in parts.items()}
    result["five_components_proxy_out_of_80"] = round(sum(parts.values()), 3)
    result["confidence_or_refusal_20"] = "not estimated"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    groups = {name: breakdown(group) for name, group in report["groups"].items()}
    print(json.dumps({
        "groups": groups,
        "note": ("仅用带金标训练集的本地代理值代入公开权重；关键事实采用过严的字面启发式，"
                 "语义用本地 BGE 余弦，结构 15 分人为采用边 F1 与完整链匹配各半，"
                 "置信度/拒答 20 分未估。以上不是平台分数，也不能与不同题目的盲测分数直接比较。"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
