"""TRAIN-only audit of task-4 retrieval and evidence-chain coverage.

This script never reads the hidden-test set or sends material to a model.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import run_baselines as baseline
import task4_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--output", type=Path,
                        default=core.REPO_ROOT / "outputs" / "research_retrieval_audit.json")
    parser.add_argument("--max-packs", type=int, default=0,
                        help="0 means all; otherwise sample pack groups with seed 20261004")
    parser.add_argument("--embedding-model", type=Path,
                        help="Optional local BGE model for hybrid ranking")
    args = parser.parse_args()

    samples = core.discover_samples(args.train_root, limit=0)
    if args.max_packs:
        packs = sorted({sample.pack for sample in samples})
        chosen = set(random.Random(20261004).sample(
            packs, min(args.max_packs, len(packs))))
        samples = [sample for sample in samples if sample.pack in chosen]
    encoder = baseline.BertEncoder(args.embedding_model) if args.embedding_model else None
    gold = {}
    for path in args.train_root.rglob("gold/问答对_答案.json"):
        for item in core.read_json(path):
            gold[item["sample_id"]] = item

    totals = Counter()
    groups = defaultdict(Counter)
    cached_packs = {}
    examples = []
    for sample in samples:
        if sample.pack not in cached_packs:
            cached_packs[sample.pack] = baseline.read_pack(sample.pack)
        pack = cached_packs[sample.pack]
        ranking = baseline.rank_candidates(sample.question, pack, encoder,
                                           hybrid=encoder is not None)
        ranked_ids = [candidate.event_id for candidate, _ in ranking]
        chains = [chain for chain in gold[sample.sample_id].get("evidence_chains", [])
                  if chain]
        if not chains:
            continue
        explicit = set(re.findall(r"D\d{3,}", sample.question))
        explicit = explicit.intersection(ranked_ids)
        explicit_gold_recall = (max(len(explicit.intersection(chain)) / len(explicit)
                                    for chain in chains) if explicit else None)
        best_recall = {}
        for k in (1, 2, 3, 5):
            selected = set(ranked_ids[:k])
            best_recall[k] = max(len(selected.intersection(chain)) / len(chain)
                                 for chain in chains)
        anchored = set(ranked_ids[:3]) | (explicit & set(ranked_ids))
        anchored_recall = max(len(anchored.intersection(chain)) / len(chain)
                              for chain in chains)
        shortest = min(map(len, chains))
        for group in (totals, groups[sample.question_type]):
            group["questions"] += 1
            group["explicit_ids"] += bool(explicit)
            if explicit_gold_recall is not None:
                group["explicit_gold_recall_sum"] += explicit_gold_recall
                group["explicit_fully_in_gold"] += explicit_gold_recall == 1
            group["shortest_chain_nodes_sum"] += shortest
            group["multi_hop"] += shortest >= 3
            group["top3_with_anchors_size_sum"] += len(anchored)
            group["top3_with_anchors_recall_sum"] += anchored_recall
            group["top3_with_anchors_full_chain"] += anchored_recall == 1
            for k, recall in best_recall.items():
                group[f"top{k}_recall_sum"] += recall
                group[f"top{k}_full_chain"] += recall == 1
        if best_recall[3] < 0.5 and len(examples) < 12:
            examples.append({"sample_id": sample.sample_id,
                             "question_type": sample.question_type,
                             "question": sample.question,
                             "top3": ranked_ids[:3],
                             "gold_chains": chains[:2]})

    def render(group: Counter) -> dict:
        n = group["questions"]
        return {"questions_with_nonempty_chain": n,
                "explicit_id_rate": round(group["explicit_ids"] / n, 4),
                "explicit_ids_fully_in_a_gold_chain_rate": (
                    round(group["explicit_fully_in_gold"] / group["explicit_ids"], 4)
                    if group["explicit_ids"] else None),
                "explicit_ids_mean_best_gold_recall": (
                    round(group["explicit_gold_recall_sum"] / group["explicit_ids"], 4)
                    if group["explicit_ids"] else None),
                "mean_shortest_chain_nodes": round(group["shortest_chain_nodes_sum"] / n, 3),
                "multi_hop_rate": round(group["multi_hop"] / n, 4),
                "top3_with_anchors_mean_size": round(group["top3_with_anchors_size_sum"] / n, 3),
                "top3_with_anchors_mean_best_gold_recall": round(group["top3_with_anchors_recall_sum"] / n, 4),
                "top3_with_anchors_full_gold_chain_rate": round(group["top3_with_anchors_full_chain"] / n, 4),
                **{f"top{k}_mean_best_gold_recall": round(group[f"top{k}_recall_sum"] / n, 4)
                   for k in (1, 2, 3, 5)},
                **{f"top{k}_full_gold_chain_rate": round(group[f"top{k}_full_chain"] / n, 4)
                   for k in (1, 2, 3, 5)}}

    result = {"scope": "training set only", "samples": len(samples),
              "packs": len(cached_packs), "all": render(totals),
              "retrieval_method": "BM25+BGE" if encoder else "BM25",
              "by_question_type": {name: render(group)
                                   for name, group in sorted(groups.items())},
              "low_recall_examples": examples,
              "definition": "For each k, compare the top-k BM25 event IDs with every nonempty gold chain and take the best node recall. This is an optimistic retrieval upper bound, not answer accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("samples", "packs", "all")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
