"""TRAIN-only paired metrics for path/answer research; all are local proxies."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import evaluate_baselines as basic
import task4_core as core


def evaluate(path: Path, train_root: Path, embedding_model: str) -> dict:
    from sentence_transformers import SentenceTransformer

    records = core.read_json(path)
    if not records:
        raise ValueError("empty paired output")
    gold = {}
    for gold_path in train_root.rglob("gold/问答对_答案.json"):
        for item in core.read_json(gold_path):
            if item["sample_id"] in gold:
                raise ValueError("duplicate gold ID")
            gold[item["sample_id"]] = item
    encoder = SentenceTransformer(embedding_model, device="cpu")
    cache = {}

    def embed(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = np.asarray(encoder.encode(text, normalize_embeddings=True,
                                                    show_progress_bar=False))
        return cache[text]

    rows = []
    for record in records:
        item = gold[record["sample_id"]]
        references = basic.answer_references(item)
        facts = basic.facts(item, "required_facts")
        alternatives = item.get("evidence_chains") or [[]]
        truth_refusal = (all(not chain for chain in alternatives)
                         and str(item.get("answers", "")).strip() == "无法确定")
        variants = {}
        for arm in ("direct", "guided", "locked"):
            if arm not in record:
                continue
            prediction = record[arm]
            chain = prediction["evidence_chain"]
            edges = [f"{a}>{b}" for a, b in zip(chain, chain[1:])]
            semantic = max((float(embed(prediction["answer"]) @ embed(ref))
                            for ref in references), default=None)
            fact_scores = [float(embed(prediction["answer"]) @ embed(fact))
                           for fact in facts if fact]
            variants[arm] = {
                "answer_char_f1": max((basic.char_f1(prediction["answer"], ref)
                                       for ref in references), default=0.0),
                "answer_semantic_cosine": semantic,
                "required_fact_mean_cosine": (sum(fact_scores) / len(fact_scores)
                                              if fact_scores else None),
                "required_fact_coverage_at_0_70":
                    (sum(value >= .70 for value in fact_scores) / len(fact_scores)
                     if fact_scores else None),
                "chain_node_f1": max(basic.set_f1(chain, gold_chain)
                                     for gold_chain in alternatives),
                "chain_edge_f1": max(basic.set_f1(edges, [f"{a}>{b}" for a, b in
                                                            zip(gold_chain, gold_chain[1:])])
                                     for gold_chain in alternatives),
                "chain_exact": float(chain in alternatives),
                "refusal_correct": float((prediction["answer"] == "无法确定") == truth_refusal),
                "refusal": float(prediction["answer"] == "无法确定"),
                "nonrefusal_confidence": prediction["confidence"],
            }
        rows.append({"sample_id": record["sample_id"], "view": record["view"],
                     "question_type": record["question_type"], "metrics": variants,
                     "guided_changed_chain": record["direct"]["evidence_chain"] !=
                     record["guided"]["evidence_chain"],
                     "locked_changed_answer": (record["guided"]["answer"] !=
                                               record["locked"]["answer"]
                                               if "locked" in record else None)})

    def summarize(subset: list[dict], arm: str) -> dict:
        keys = subset[0]["metrics"][arm]
        result = {"samples": len(subset)}
        for key in keys:
            vals = [row["metrics"][arm][key] for row in subset
                    if row["metrics"][arm][key] is not None]
            result[key] = round(sum(vals) / len(vals), 4) if vals else None
            if key == "required_fact_coverage_at_0_70":
                result["samples_with_required_facts"] = len(vals)
        return result

    grouped = {"all": rows, **{view: [row for row in rows if row["view"] == view]
                              for view in "ABC"}}
    summary = {name: {arm: summarize(subset, arm) for arm in
                      ("direct", "guided", "locked")
                      if all(arm in row["metrics"] for row in subset)}
               for name, subset in grouped.items() if subset}
    costs = {}
    for name in ("direct_usage", "guided_usage", "rewrite_usage"):
        aggregate = Counter()
        calls = 0
        for record in records:
            usage = record.get(name, {})
            if not usage.get("skipped_refusal"):
                calls += usage.get("request_count", int(bool(usage)))
                aggregate.update({k: v for k, v in usage.items()
                                  if k in {"prompt_tokens", "completion_tokens", "total_tokens",
                                           "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                                           "truncated_calls"}
                                  and isinstance(v, int)})
        # Official Flash rates checked 2026-10-05; range covers off-peak/peak.
        hit = aggregate["prompt_cache_hit_tokens"]
        miss = aggregate["prompt_cache_miss_tokens"]
        output = aggregate["completion_tokens"]
        off_peak_usd = (hit * .003 + miss * .15 + output * .6) / 1_000_000
        costs[name] = {"requests_at_least": calls, **dict(aggregate),
                       "estimated_usd_off_peak": round(off_peak_usd, 5),
                       "estimated_usd_peak": round(2 * off_peak_usd, 5)}
    result = {"samples": len(rows), "model": records[0]["model"],
              "semantic_model": embedding_model,
              "fact_coverage_rule": "BGE cosine >= 0.70 per required_fact; exploratory local proxy, not official scorer",
              "confidence_note": "Mean stated confidence is descriptive, not calibrated correctness or the official confidence score.",
              "cost_note": "USD range uses DeepSeek Flash list prices and reported token usage; earlier truncated calls without usage make this a lower bound. No official per-item scoring formula assumed.",
              "summary": summary, "cost_tokens": costs,
              "input_chars": {arm: sum(record[f"{arm}_input_chars"] for record in records)
                              for arm in ("direct", "guided")},
              "elapsed_seconds": round(sum(record["elapsed_seconds"] for record in records), 2),
              "rows": rows}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("Paired API metrics must be written under ignored outputs/")
    report = evaluate(args.paired, args.train_root, args.embedding_model)
    core.atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
