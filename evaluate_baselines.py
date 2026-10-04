"""Transparent TRAIN-only diagnostics; the hidden official scorer is unavailable."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import task4_core as core


def char_f1(prediction: str, reference: str) -> float:
    def chars(value: str) -> Counter:
        return Counter(c for c in value.lower() if c.isalnum())
    left, right = chars(prediction), chars(reference)
    if not left or not right:
        return float(not left and not right)
    return 2 * sum((left & right).values()) / (sum(left.values()) + sum(right.values()))


def set_f1(prediction: list[str], reference: list[str]) -> float:
    if not prediction and not reference:
        return 1.0
    return 2 * len(set(prediction) & set(reference)) / (len(prediction) + len(reference))


def answer_references(item: dict) -> list[str]:
    value = item.get("answers", item.get("answer", ""))
    return [value] if isinstance(value, str) else [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def facts(item: dict, kind: str) -> list[str]:
    values = (item.get("answer_facts") or {}).get(kind) or []
    return [x if isinstance(x, str) else str(x.get("fact", x.get("text", "")))
            for x in values if isinstance(x, (str, dict))]


def evaluate(predictions: Path, train_root: Path, embedding_model: Path | None = None) -> dict:
    sample_map = {sample.sample_id: sample for sample in core.discover_samples(train_root, limit=0)}
    gold: dict[str, dict] = {}
    for path in train_root.rglob("gold/问答对_答案.json"):
        for item in core.read_json(path):
            if item["sample_id"] in gold:
                raise ValueError(f"训练集重复金标 ID：{item['sample_id']}")
            gold[item["sample_id"]] = item
    records = core.read_json(predictions)
    if not isinstance(records, list) or not records:
        raise ValueError("预测文件须为非空 JSON 数组")
    encoder = None
    if embedding_model is not None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(embedding_model), device="cpu")
    grouped = defaultdict(list)
    seen = set()
    for record in records:
        sample_id = record.get("sample_id") if isinstance(record, dict) else None
        if sample_id not in gold or sample_id not in sample_map:
            raise ValueError(f"预测不属于带金标的训练集：{sample_id}")
        if sample_id in seen:
            raise ValueError(f"预测包含重复 ID：{sample_id}")
        seen.add(sample_id)
        sample = sample_map[sample_id]
        _, allowed, _ = core.build_input(sample)
        parsed = core.parse_prediction(json.dumps(record, ensure_ascii=False), sample, allowed)
        if parsed != record:
            raise ValueError(f"预测提交字段无效：{sample_id}")
        item = gold[sample_id]
        chains = item.get("evidence_chains") or [[]]
        actual = record["evidence_chain"]
        references = answer_references(item)
        required = facts(item, "required_facts")
        forbidden = facts(item, "forbidden_facts")
        def edges(chain: list[str]) -> list[str]:
            return [f"{a}>{b}" for a, b in zip(chain, chain[1:])]
        row = {
            "answer_char_f1": max((char_f1(record["answer"], ref) for ref in references), default=0.0),
            "chain_exact": float(actual in chains),
            "chain_event_f1": max(set_f1(actual, chain) for chain in chains),
            "chain_edge_f1": max(set_f1(edges(actual), edges(chain)) for chain in chains),
            "refusal_correct": float((record["answer"] == "无法确定") == all(not chain for chain in chains)),
            "required_fact_coverage": (sum(fact in record["answer"] or char_f1(record["answer"], fact) >= 0.5
                                           for fact in required) / len(required) if required else None),
            "forbidden_fact_hits": sum(fact in record["answer"] for fact in forbidden if fact),
            "confidence": record["confidence"],
            "gold_confidence_level": item.get("confidence_level"),
        }
        if encoder is not None and references:
            import numpy as np
            vectors = encoder.encode([record["answer"], *references], normalize_embeddings=True,
                                     show_progress_bar=False)
            row["answer_semantic_cosine"] = float(max(np.asarray(vectors[1:]) @ vectors[0]))
        grouped["all"].append(row)
        grouped[f"type:{sample.question_type}"].append(row)

    def summarize(rows: list[dict]) -> dict:
        result = {"samples": len(rows)}
        for key in ("answer_char_f1", "answer_semantic_cosine", "chain_exact", "chain_event_f1",
                    "chain_edge_f1", "refusal_correct", "required_fact_coverage"):
            values = [row[key] for row in rows if row.get(key) is not None]
            if values:
                result[key] = round(sum(values) / len(values), 4)
                if key == "required_fact_coverage":
                    result["samples_with_required_facts"] = len(values)
        result["forbidden_fact_hits"] = sum(row["forbidden_fact_hits"] for row in rows)
        result["refusals"] = sum(row["confidence"] is None for row in rows)
        confidence_groups = defaultdict(list)
        for row in rows:
            if row["confidence"] is not None and row["gold_confidence_level"] is not None:
                confidence_groups[str(row["gold_confidence_level"])].append(row["confidence"])
        if confidence_groups:
            result["mean_confidence_by_gold_level"] = {
                level: round(sum(values) / len(values), 4)
                for level, values in sorted(confidence_groups.items())}
        return result

    return {"groups": {name: summarize(rows) for name, rows in sorted(grouped.items())},
            "note": "仅带金标训练集上的本地诊断；字符 F1、可选 BERT 余弦及证据边指标均不是官方公式。测试集无金标，无法推算官方得分。"}


def inspect_test(predictions: Path, test_root: Path) -> dict:
    """Validate the entire blind-test submission and report non-score diagnostics."""
    samples = core.discover_samples(test_root, limit=0)
    core.validate_file(predictions, samples, 80000)
    sample_map = {sample.sample_id: sample for sample in samples}
    grouped = defaultdict(list)
    for record in core.read_json(predictions):
        sample = sample_map[record["sample_id"]]
        _, _, edges = core.build_input(sample)
        chain = record["evidence_chain"]
        row = {
            "refusal": record["answer"] == "无法确定",
            "answer_chars": len(record["answer"]),
            "chain_nodes": len(chain),
            "unsupported_adjacent_edges": (sum((a, b) not in edges for a, b in zip(chain, chain[1:]))
                                           if edges else None),
        }
        grouped[sample.track].append(row)
    return {
        "samples": len(samples),
        "by_track": {track: {
            "samples": len(rows),
            "refusals": sum(row["refusal"] for row in rows),
            "mean_answer_chars": round(sum(row["answer_chars"] for row in rows) / len(rows), 2),
            "mean_chain_nodes": round(sum(row["chain_nodes"] for row in rows) / len(rows), 2),
            "unsupported_adjacent_edges": sum(row["unsupported_adjacent_edges"] or 0 for row in rows),
        } for track, rows in sorted(grouped.items())},
        "note": "盲测集只有结构与分布诊断；无标准答案，以上数字不能换算为官方得分或准确率。",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-root", type=Path)
    source.add_argument("--test-root", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    args = parser.parse_args()
    result = (evaluate(args.predictions, args.train_root, args.embedding_model)
              if args.train_root else inspect_test(args.predictions, args.test_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
