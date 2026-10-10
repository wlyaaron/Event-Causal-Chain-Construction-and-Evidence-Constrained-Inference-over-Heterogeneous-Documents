"""Audit frozen V3 candidate ranking against V2 logic, fit only.

Candidates are computed from visible inputs before the corresponding gold is read.
No model outputs are used.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task4_core as core
from experiments.audit_a_candidate_targeted_20261010 import (
    SPLIT, TRAIN, OUT_DIR, group_for,
)
from experiments.task4_a_candidate_pool import (
    enumerate_graph_paths, rank_graph_paths, rank_graph_paths_targeted,
    targeted_candidate_intent,
)


def main() -> None:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    changes = []
    for pack_name in core.read_json(SPLIT)["splits"]["fit"]:
        pack = TRAIN / pack_name
        documents, events, edges = core.load_pack(pack)
        paths = enumerate_graph_paths(events, edges)
        event_ids = {event["event_id"] for event in events}
        questions = core.read_json(pack / "问题.json")
        proposals = {}
        for question in questions:
            intent, _ = targeted_candidate_intent(
                question["question"], question["question_type"], event_ids)
            v3 = rank_graph_paths_targeted(
                question["question"], question["question_type"], events, edges,
                documents, paths, limit=5)
            v2 = (rank_graph_paths(question["question"], events, documents,
                                   paths, limit=5)
                  if intent in {"temporal_causal_pair", "unspecified_direct"} else v3)
            proposals[question["sample_id"]] = (intent, v2, v3)
        gold = {item["sample_id"]: item for item in
                core.read_json(pack / "gold" / "问答对_答案.json")}
        for question in questions:
            sample_id = question["sample_id"]
            intact = {tuple(chain) for chain in
                      gold[sample_id].get("evidence_chains", []) if chain}
            if not intact:
                continue
            intent, v2, v3 = proposals[sample_id]
            v2_hit = bool(intact & set(v2))
            v3_hit = bool(intact & set(v3))
            group = group_for(question)
            counter = groups[group]
            counter["n"] += 1
            counter["v2_hit"] += v2_hit
            counter["v3_hit"] += v3_hit
            counter["gains"] += v3_hit and not v2_hit
            counter["losses"] += v2_hit and not v3_hit
            if intent:
                counter[f"intent_{intent}"] += 1
            if v3_hit != v2_hit:
                changes.append({"sample_id": sample_id, "pack": pack_name,
                                "group": group, "intent": intent,
                                "v2_hit": v2_hit, "v3_hit": v3_hit})
    output = {"scope": "fit only, simulated A view",
              "comparison": "V2 vs frozen V3, exact any complete gold chain at top5",
              "groups": {k: dict(v) for k, v in sorted(groups.items())},
              "changes": changes}
    path = OUT_DIR / "a_candidate_v3_fit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({"path": str(path), "groups": output["groups"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
