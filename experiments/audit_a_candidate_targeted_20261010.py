"""Paired, gold-blind candidate audit on one pack-isolated training split.

All candidates for a pack are fixed from question and materials before its
gold file is opened. Fit is for development; validation is a one-time check.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import task4_core as core  # noqa: E402
from experiments.summarize_a_candidate_scope_20261010 import PATH_CUE  # noqa: E402
from experiments.task4_a_candidate_pool import (  # noqa: E402
    enumerate_graph_paths,
    rank_graph_paths,
    rank_graph_paths_targeted,
    targeted_candidate_intent,
)


TRAIN = ROOT / "数据集" / "训练集"
SPLIT = ROOT / "experiments" / "finetune_split_20261005.json"
OUT_DIR = ROOT / "experiments" / "research_audit_20261010"
KS = (5, 10, 16)


def group_for(question: dict) -> str:
    if question["question_type"] != "retrospective":
        return question["question_type"]
    return ("explicit_path_retrospective" if PATH_CUE.search(question["question"])
            else "other_retrospective")


def hits(ranked: list[tuple[str, ...]],
         gold: list[list[str]]) -> dict[str, bool]:
    return {str(k): bool({tuple(chain) for chain in gold}
                         & set(ranked[:k])) for k in KS}


def audit(split_name: str) -> dict:
    pack_names = core.read_json(SPLIT)["splits"][split_name]
    rows: list[dict] = []
    groups: dict[str, Counter[str]] = defaultdict(Counter)

    for pack_name in pack_names:
        pack = TRAIN / pack_name
        documents, events, edges = core.load_pack(pack)
        if events is None or edges is None:
            raise ValueError(f"Missing A-view inputs: {pack_name}")
        paths = enumerate_graph_paths(events, edges)
        path_set = set(paths)
        event_ids = {event["event_id"] for event in events}
        questions = core.read_json(pack / "问题.json")
        proposals = {}
        for question in questions:
            text = question["question"]
            question_type = question["question_type"]
            intent, anchor = targeted_candidate_intent(
                text, question_type, event_ids)
            old = rank_graph_paths(text, events, documents, paths, limit=max(KS))
            new = (rank_graph_paths_targeted(
                text, question_type, events, edges, documents, paths,
                limit=max(KS)) if intent else old)
            proposals[question["sample_id"]] = (old, new, intent, anchor)

        # Gold is opened only after predictions for every question in this
        # material pack have been generated.
        gold_by_id = {item["sample_id"]: item for item in
                      core.read_json(pack / "gold" / "问答对_答案.json")}
        for question in questions:
            sample_id = question["sample_id"]
            old, new, intent, anchor = proposals[sample_id]
            gold = [chain for chain in gold_by_id[sample_id].get("evidence_chains", [])
                    if isinstance(chain, list) and chain]
            group = group_for(question)
            count = groups[group]
            count["all"] += 1
            if not gold:
                count["empty_gold"] += 1
                continue
            count["nonempty_gold"] += 1
            old_hits = hits(old, gold)
            new_hits = hits(new, gold)
            raw_hit = any(tuple(chain) in path_set for chain in gold)
            count["raw_hit"] += raw_hit
            if intent:
                count[f"intent_{intent}"] += 1
            for k in KS:
                key = str(k)
                count[f"old_{key}"] += old_hits[key]
                count[f"new_{key}"] += new_hits[key]
                count[f"gain_{key}"] += new_hits[key] and not old_hits[key]
                count[f"loss_{key}"] += old_hits[key] and not new_hits[key]
            rows.append({
                "sample_id": sample_id,
                "pack": pack_name,
                "group": group,
                "intent": intent,
                "anchor": anchor,
                "raw_hit": raw_hit,
                "old_hit": old_hits,
                "new_hit": new_hits,
            })

    return {
        "scope": f"{split_name} packs, simulated A view",
        "pack_count": len(pack_names),
        "metric": "exact match to any intact, nonempty original gold chain",
        "candidate_inputs": ["question", "question_type", "documents", "events", "edges"],
        "groups": {key: dict(sorted(value.items())) for key, value in sorted(groups.items())},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("fit", "validation"))
    args = parser.parse_args()
    result = audit(args.split)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"a_candidate_targeted_{args.split}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"path": str(output_path), "pack_count": result["pack_count"],
                      "groups": result["groups"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
