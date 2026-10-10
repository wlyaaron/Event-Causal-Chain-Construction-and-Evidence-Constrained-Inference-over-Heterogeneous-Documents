"""Fit-only, no-model audit of complete gold-chain recall for A-view candidates.

Candidate generation and ranking use only question, documents, events and
visible causal relations. Gold is loaded afterwards solely for scoring.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import task4_core as core  # noqa: E402
import task4_workflow as workflow  # noqa: E402
from experiments.task4_a_candidate_pool import (  # noqa: E402
    enumerate_graph_paths,
    rank_graph_paths,
)


TRAIN = ROOT / "数据集" / "训练集"
SPLIT = ROOT / "experiments" / "finetune_split_20261005.json"
OUT = ROOT / "experiments" / "research_audit_20261010" / "a_candidate_pool_fit.json"
KS = (5, 10, 16)


def nonmonotonic(chain: list[str]) -> bool:
    return any(int(after[1:]) < int(before[1:])
               for before, after in zip(chain, chain[1:]))


def covered(paths: set[tuple[str, ...]], gold: list[list[str]]) -> bool:
    return any(tuple(chain) in paths for chain in gold)


def reversal_census(splits: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Describe all training gold; never feed these counts to ranking."""
    result: dict[str, dict[str, int]] = {}
    for split_name, pack_names in splits.items():
        count: Counter[str] = Counter()
        for pack_name in pack_names:
            pack = TRAIN / pack_name
            graph_edges = {
                (edge["cause_event_id"],
                 edge.get("result_event_id", edge.get("effect_event_id")))
                for edge in core.read_json(pack / "事件因果关系列表.json")
            }
            for gold in core.read_json(pack / "gold" / "问答对_答案.json"):
                chains = [chain for chain in gold.get("evidence_chains", [])
                          if isinstance(chain, list) and len(chain) > 1]
                if not chains:
                    continue
                count["multi_questions"] += 1
                count["nonmonotonic_questions"] += any(nonmonotonic(chain)
                                                       for chain in chains)
                for chain in chains:
                    count["multi_alternatives"] += 1
                    if nonmonotonic(chain):
                        count["nonmonotonic_alternatives"] += 1
                        count["nonmonotonic_graph_full_alternatives"] += all(
                            (before, after) in graph_edges
                            for before, after in zip(chain, chain[1:])
                        )
        result[split_name] = dict(sorted(count.items()))
    return result


def main() -> None:
    splits = core.read_json(SPLIT)["splits"]
    fit_packs = splits["fit"]
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    graph_path_counts: list[int] = []

    for pack_name in fit_packs:
        pack = TRAIN / pack_name
        documents, events, edges = core.load_pack(pack)
        if events is None or edges is None:
            raise ValueError(f"A-view pack lacks events or graph: {pack_name}")
        raw_paths = enumerate_graph_paths(events, edges)
        raw_set = set(raw_paths)
        graph_path_counts.append(len(raw_paths))
        questions = core.read_json(pack / "问题.json")
        proposals: dict[str, tuple[list[tuple[str, ...]], list[list[str]]]] = {}
        for item in questions:
            sample_id = item["sample_id"]
            new_ranked = rank_graph_paths(item["question"], events, documents,
                                          raw_paths, limit=max(KS))
            old_sample = core.Sample(pack, "train", sample_id,
                                     item["question_type"], item["question"])
            old_paths = workflow.question_conditioned_paths(old_sample, limit=5)
            proposals[sample_id] = (new_ranked, old_paths)

        # Read gold only after every candidate for this pack has been frozen.
        gold_by_id = {item["sample_id"]: item for item in
                      core.read_json(pack / "gold" / "问答对_答案.json")}

        for item in questions:
            sample_id = item["sample_id"]
            gold_chains = [chain for chain in gold_by_id[sample_id].get("evidence_chains", [])
                           if isinstance(chain, list) and chain]
            if not gold_chains:
                counts["empty_gold_chain"] += 1
                continue

            counts["nonempty_gold_chain"] += 1
            is_multi = any(len(chain) > 1 for chain in gold_chains)
            is_nonmonotonic = any(nonmonotonic(chain) for chain in gold_chains)
            if is_multi:
                counts["multi_gold_question"] += 1
            if is_nonmonotonic:
                counts["nonmonotonic_gold_question"] += 1
            new_ranked, old_paths = proposals[sample_id]
            new_hits = {str(k): covered(set(new_ranked[:k]), gold_chains) for k in KS}
            raw_hit = covered(raw_set, gold_chains)
            old_hit = covered({tuple(path) for path in old_paths}, gold_chains)

            counts["raw_hit"] += raw_hit
            counts["old_top5_hit"] += old_hit
            for k in KS:
                counts[f"new_top{k}_hit"] += new_hits[str(k)]
            if is_multi:
                counts["multi_raw_hit"] += raw_hit
                counts["multi_old_top5_hit"] += old_hit
                for k in KS:
                    counts[f"multi_new_top{k}_hit"] += new_hits[str(k)]
            if is_nonmonotonic:
                counts["nonmonotonic_raw_hit"] += raw_hit
                counts["nonmonotonic_old_top5_hit"] += old_hit
                for k in KS:
                    counts[f"nonmonotonic_new_top{k}_hit"] += new_hits[str(k)]

            rows.append({
                "sample_id": sample_id,
                "pack": pack_name,
                "question_type": item["question_type"],
                "reasoning_type": item.get("reasoning_type"),
                "gold_chain_lengths": [len(chain) for chain in gold_chains],
                "nonmonotonic_gold": is_nonmonotonic,
                "raw_graph_hit": raw_hit,
                "old_top5_hit": old_hit,
                "new_top_k_hit": new_hits,
                "raw_candidate_count": len(raw_paths),
            })

    counts["fit_packs"] = len(fit_packs)
    counts["all_questions"] = counts["empty_gold_chain"] + counts["nonempty_gold_chain"]
    summary = {
        "scope": "fit packs, simulated A view for every fit question",
        "method": "all singleton/simple directed graph paths; BM25 and question-role ranking",
        "candidate_input_excludes": ["gold", "reasoning_type", "model_output", "blind_test"],
        "counts": dict(sorted(counts.items())),
        "graph_paths_per_pack": {
            "mean": round(sum(graph_path_counts) / len(graph_path_counts), 2),
            "max": max(graph_path_counts),
        },
        "descriptive_reversal_census_by_split": reversal_census(splits),
        "limitations": [
            "Raw graph paths cannot cover complete gold chains with a missing visible edge.",
            "An exact gold alternative must appear intact; alternatives are never combined.",
            "Candidate recall is an oracle diagnostic, not model answer accuracy or official score.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                              indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
