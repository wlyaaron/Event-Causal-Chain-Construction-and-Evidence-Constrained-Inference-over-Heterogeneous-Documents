"""Gold-only, pack-isolated validation/holdout proxies for offline SFT runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import task4_core as core
from evaluate_baselines import answer_references, char_f1, set_f1, facts


def score_row(prediction: dict, gold: dict) -> dict:
    answer = prediction["answer"]
    chain = prediction["evidence_chain"]
    alternatives = gold.get("evidence_chains") or [[]]
    references = answer_references(gold)
    required = facts(gold, "required_facts")
    def edges(nodes):
        return [f"{left}>{right}" for left, right in zip(nodes, nodes[1:])]
    return {
        "answer_char_f1": max((char_f1(answer, ref) for ref in references), default=0.0),
        "chain_event_f1": max(set_f1(chain, ref) for ref in alternatives),
        "chain_edge_f1": max(set_f1(edges(chain), edges(ref)) for ref in alternatives),
        "chain_exact": float(chain in alternatives),
        "refusal_correct": float((answer == "无法确定") ==
                                 all(not ref for ref in alternatives)),
        "required_fact_coverage": (sum(fact in answer or char_f1(answer, fact) >= .5
                                       for fact in required) / len(required)
                                   if required else None),
    }


def summarize(rows: list[dict]) -> dict:
    result = {"samples": len(rows)}
    for metric in ("answer_char_f1", "answer_semantic_cosine", "required_fact_coverage",
                   "chain_event_f1", "chain_edge_f1", "chain_exact", "refusal_correct"):
        values = [row[metric] for row in rows if row.get(metric) is not None]
        result[metric] = round(sum(values) / len(values), 4) if values else None
        if metric == "required_fact_coverage":
            result["samples_with_required_facts"] = len(values)
    result["invalid_predictions"] = sum(not row["valid"] for row in rows)
    result["refusals"] = sum(row["refusal"] for row in rows)
    result["total_input_tokens"] = sum(row["input_tokens"] for row in rows)
    result["total_output_tokens"] = sum(row["output_tokens"] for row in rows)
    result["total_latency_seconds"] = round(sum(row["latency_seconds"] for row in rows), 2)
    result["mean_input_tokens"] = round(sum(row["input_tokens"] for row in rows) / len(rows), 1)
    result["mean_output_tokens"] = round(sum(row["output_tokens"] for row in rows) / len(rows), 1)
    result["mean_latency_seconds"] = round(sum(row["latency_seconds"] for row in rows) / len(rows), 2)
    if result["answer_semantic_cosine"] is not None:
        result["five_components_proxy_out_of_80"] = round(
            25 * (result["required_fact_coverage"] or 0) +
            10 * result["answer_char_f1"] +
            5 * max(0, min(1, result["answer_semantic_cosine"])) +
            25 * result["chain_event_f1"] +
            7.5 * (result["chain_edge_f1"] + result["chain_exact"]), 3)
    else:
        result["five_components_proxy_out_of_80"] = None
    return result


def semantic_vectors(texts: list[str], embedding_model: Path,
                     stats: dict | None = None):
    """BGE small zh v1.5: CLS pooling, L2 norm, 512-token input cap."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    # The current container has only 0.5 CPU core; PyTorch's host-wide
    # default created hundreds of competing threads in a CPU smoke run.
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model, local_files_only=True)
    model = AutoModel.from_pretrained(embedding_model, local_files_only=True).to("cpu")
    model.eval()
    vectors = []
    overlength = 0
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            chunk = texts[start:start + 16]
            lengths = tokenizer(chunk, add_special_tokens=True,
                                truncation=False)["input_ids"]
            overlength += sum(len(ids) > 512 for ids in lengths)
            batch = tokenizer(chunk, padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
            cls = model(**batch).last_hidden_state[:, 0]
            vectors.extend(torch.nn.functional.normalize(cls, p=2, dim=1).cpu().tolist())
    if stats is not None:
        stats["semantic_texts_over_512_tokens"] = overlength
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "holdout"], required=True)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ignored = (core.REPO_ROOT / "outputs").resolve()
    for path in (args.predictions, args.data_dir, args.output):
        if not path.resolve().is_relative_to(ignored):
            raise ValueError("predictions and reports must stay under ignored outputs/")
    split_rows = {}
    for view in "ABC":
        with (args.data_dir / f"{args.split}_{view}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                split_rows[(row["pack"], row["sample_id"], view)] = row
    gold = {}
    for pack in {row["pack"] for row in split_rows.values()}:
        path = args.train_root / pack / "gold" / "问答对_答案.json"
        for item in core.read_json(path):
            gold[(pack, item["sample_id"])] = item
    predictions = [json.loads(line) for line in args.predictions.open(encoding="utf-8")]
    seen = set()
    records = []
    for prediction in predictions:
        key = (prediction["pack"], prediction["sample_id"], prediction["view"])
        if key in seen or key not in split_rows:
            raise ValueError(f"duplicate or out-of-split prediction: {key}")
        seen.add(key)
        row = split_rows[key]
        if prediction.get("valid"):
            metrics = score_row(prediction, gold[key[:2]])
        else:
            metrics = {name: 0.0 for name in ("answer_char_f1", "chain_event_f1",
                                                   "chain_edge_f1", "chain_exact", "refusal_correct")}
            metrics["required_fact_coverage"] = 0.0 if facts(gold[key[:2]], "required_facts") else None
        metrics.update({"pack": key[0], "sample_id": key[1], "view": key[2],
                        "question_type": row["question_type"], "flags": row["flags"],
                        "valid": bool(prediction.get("valid")),
                        "refusal": prediction.get("answer") == "无法确定",
                        "input_tokens": int(prediction.get("input_tokens", 0)),
                        "output_tokens": int(prediction.get("output_tokens", 0)),
                        "latency_seconds": float(prediction.get("latency_seconds", 0))})
        records.append(metrics)
    if seen != set(split_rows):
        raise ValueError(f"missing {len(set(split_rows) - seen)} split/view predictions")
    semantic_stats = {}
    if args.embedding_model:
        import numpy as np
        texts = []
        reference_spans = []
        for prediction in predictions:
            key = (prediction["pack"], prediction["sample_id"])
            references = answer_references(gold[key])
            start = len(texts)
            texts.extend([str(prediction.get("answer", "")), *references])
            reference_spans.append((start, len(texts)))
        vectors = semantic_vectors(texts, args.embedding_model, semantic_stats)
        for record, (start, end) in zip(records, reference_spans):
            record["answer_semantic_cosine"] = (
                max(float(np.dot(vectors[start], vectors[index]))
                    for index in range(start + 1, end))
                if record["valid"] and end > start + 1 else 0.0)
    grouped = defaultdict(list)
    for record in records:
        view = record["view"]
        grouped[view].append(record)
        grouped[f"{view}:type:{record['question_type']}"].append(record)
        grouped[f"{view}:same_question_answer_as_fit:{'yes' if 'same_question_answer_as_fit' in record['flags'] else 'no'}"].append(record)
        grouped[f"{view}:graph_step_absent:{'yes' if 'step_absent_from_given_edges' in record['flags'] else 'no'}"].append(record)
        grouped[f"{view}:explanatory_unanswerable:{'yes' if 'unanswerable_with_evidence' in record['flags'] else 'no'}"].append(record)
        grouped[f"{view}:long_input:{'yes' if record['input_tokens'] > 8192 else 'no'}"].append(record)
    summaries = {name: summarize(rows) for name, rows in sorted(grouped.items())}
    weighted = None
    if args.embedding_model:
        weighted = round(sum(weight * summaries[view]["five_components_proxy_out_of_80"]
                             for view, weight in (("A", .6), ("B", .3), ("C", .1))), 3)
    report = {"split": args.split, "prediction_count": len(records),
              **semantic_stats,
              "groups": summaries, "weighted_five_components_proxy_out_of_80": weighted,
              "confidence_or_refusal_20": "not estimated",
              "worst_answer_examples": sorted(({"pack": row["pack"],
                                                "sample_id": row["sample_id"],
                                                "view": row["view"],
                                                "answer_char_f1": row["answer_char_f1"],
                                                "chain_exact": row["chain_exact"]}
                                               for row in records),
                                              key=lambda row: row["answer_char_f1"])[:30],
              "note": "Only organizer TRAIN gold is used. Semantic cosine and literal fact coverage, graph structure proxy and public weights are local proxies, not the hidden scorer. Missing required facts are excluded from that component mean; confidence remains uncalibrated."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"split": args.split, "predictions": len(records),
                      "views": {view: summaries[view] for view in "ABC"},
                      "weighted_proxy_out_of_80": weighted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
