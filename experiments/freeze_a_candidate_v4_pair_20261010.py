"""Freeze 50 holdout-pack A-view questions using inference inputs only.

The split is from organizer training data. This module never reads gold.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
import task4_workflow as workflow  # noqa: E402
from experiments.summarize_a_candidate_scope_20261010 import PATH_CUE  # noqa: E402
from experiments.task4_a_candidate_pool import (  # noqa: E402
    enumerate_graph_paths, rank_graph_paths_targeted, targeted_candidate_intent,
)

TRAIN = ROOT / "数据集" / "训练集"
SPLIT = ROOT / "experiments" / "finetune_split_20261005.json"
PROMPT = ROOT / "experiments" / "prompts" / "task4_merged_rules_v4_draft_20261009.txt"
OUT = ROOT / "experiments" / "a_candidate_v4_pair_holdout50_20261010.json"
QUOTAS = {"explicit_path": 15, "temporal_causal_pair": 15,
          "direct_effect": 15, "unspecified_direct": 2, "event_facts": 3}


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def category(question: dict, ids: set[str]) -> str | None:
    intent, _ = targeted_candidate_intent(question["question"],
                                           question["question_type"], ids)
    if intent in QUOTAS:
        return intent
    if question["question_type"] == "retrospective" and PATH_CUE.search(question["question"]):
        return "explicit_path"
    return None


def freeze() -> dict:
    pools = defaultdict(list)
    holdout_packs = core.read_json(SPLIT)["splits"]["holdout"]
    for pack_name in holdout_packs:
        pack = TRAIN / pack_name
        _, events, _ = core.load_pack(pack)
        ids = {event["event_id"] for event in events}
        for question in core.read_json(pack / "问题.json"):
            kind = category(question, ids)
            if kind:
                pools[kind].append((pack_name, question))
    selected = []
    used_packs = set()
    for kind in ("event_facts", "unspecified_direct", "direct_effect",
                 "temporal_causal_pair", "explicit_path"):
        quota = QUOTAS[kind]
        options = sorted(pools[kind], key=lambda item: sha(
            f"20261010|{kind}|{item[0]}|{item[1]['sample_id']}"))
        for pack_name, question in options:
            if pack_name not in used_packs:
                selected.append((kind, pack_name, question))
                used_packs.add(pack_name)
                if sum(row[0] == kind for row in selected) == quota:
                    break
        if sum(row[0] == kind for row in selected) != quota:
            chosen_ids = {row[2]["sample_id"] for row in selected}
            for pack_name, question in options:
                if question["sample_id"] in chosen_ids:
                    continue
                selected.append((kind, pack_name, question))
                chosen_ids.add(question["sample_id"])
                if sum(row[0] == kind for row in selected) == quota:
                    break
        if sum(row[0] == kind for row in selected) != quota:
            raise ValueError(f"Insufficient questions in {kind}")
    rows = []
    for kind, pack_name, question in selected:
        sample = core.Sample(TRAIN / pack_name, "train", question["sample_id"],
                             question["question_type"], question["question"])
        base, _, _ = workflow.training_view_input(sample, "A")
        payload = json.loads(base)
        if payload.pop("track") != "A":
            raise ValueError("Unexpected view")
        blind = json.dumps(payload, ensure_ascii=False)
        documents, events, edges = core.load_pack(sample.pack)
        paths = enumerate_graph_paths(events, edges)
        candidates = rank_graph_paths_targeted(
            question["question"], question["question_type"], events, edges,
            documents, paths, limit=5)
        rows.append({"sample_id": sample.sample_id, "pack": pack_name,
                     "view": "A", "question_type": sample.question_type,
                     "question": sample.question, "category": kind,
                     "input_sha256": sha(blind),
                     "candidate_paths": [list(path) for path in candidates]})
    if len(rows) != 50 or len({row["sample_id"] for row in rows}) != 50:
        raise ValueError("Expected 50 distinct holdout questions")
    return {"seed": 20261010, "scope": "organizer training holdout packages; simulated A view",
            "selection": "input-only category quotas, hash ordering, prefer distinct packs",
            "gold_used_for_selection": False, "quotas": QUOTAS,
            "candidate_pool_sha256": sha((ROOT / "experiments" /
                                           "task4_a_candidate_pool.py").read_bytes()),
            "v4_prompt_sha256": sha(PROMPT.read_bytes()), "rows": rows}


def main() -> None:
    result = freeze()
    if OUT.exists() and core.read_json(OUT) != result:
        raise ValueError("Frozen manifest differs; refusing overwrite")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"path": str(OUT), "n": len(result["rows"]),
                      "distinct_packs": len({r["pack"] for r in result["rows"]}),
                      "categories": dict(Counter(r["category"] for r in result["rows"])),
                      "gold_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
