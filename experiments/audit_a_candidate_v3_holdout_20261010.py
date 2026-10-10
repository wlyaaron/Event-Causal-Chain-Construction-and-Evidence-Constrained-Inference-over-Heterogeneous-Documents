"""One-time holdout gold audit after the frozen 50-question API run completes."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
from experiments.audit_a_candidate_targeted_20261010 import (  # noqa: E402
    SPLIT, TRAIN, OUT_DIR, group_for,
)
from experiments.task4_a_candidate_pool import (  # noqa: E402
    enumerate_graph_paths, rank_graph_paths, rank_graph_paths_targeted,
    targeted_candidate_intent,
)

ATTEMPTS = ROOT / "outputs" / "a_candidate_v4_pair_holdout50_20261010" / "attempts.jsonl"


def main() -> None:
    if not ATTEMPTS.exists() or len(ATTEMPTS.read_text(encoding="utf-8").splitlines()) != 100:
        raise RuntimeError("Holdout gold audit requires all 100 frozen model attempts first")
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    changed = []
    for pack_name in core.read_json(SPLIT)["splits"]["holdout"]:
        pack = TRAIN / pack_name
        documents, events, edges = core.load_pack(pack)
        paths = enumerate_graph_paths(events, edges)
        ids = {event["event_id"] for event in events}
        questions = core.read_json(pack / "问题.json")
        proposals = {}
        for question in questions:
            text = question["question"]
            question_type = question["question_type"]
            intent, _ = targeted_candidate_intent(text, question_type, ids)
            v3 = rank_graph_paths_targeted(text, question_type, events, edges,
                                            documents, paths, limit=5)
            v2 = (rank_graph_paths(text, events, documents, paths, limit=5)
                  if intent in {"temporal_causal_pair", "unspecified_direct"} else v3)
            proposals[question["sample_id"]] = (intent, v2, v3)
        gold = {item["sample_id"]: item for item in
                core.read_json(pack / "gold" / "问答对_答案.json")}
        for question in questions:
            sid = question["sample_id"]
            intact = {tuple(chain) for chain in
                      gold[sid].get("evidence_chains", []) if chain}
            if not intact:
                continue
            intent, v2, v3 = proposals[sid]
            old_hit = bool(intact & set(v2))
            new_hit = bool(intact & set(v3))
            counter = groups[group_for(question)]
            counter["n"] += 1
            counter["v2_hit"] += old_hit
            counter["v3_hit"] += new_hit
            counter["gain"] += new_hit and not old_hit
            counter["loss"] += old_hit and not new_hit
            if intent:
                counter[f"intent_{intent}"] += 1
            if new_hit != old_hit:
                changed.append({"sample_id": sid, "pack": pack_name,
                                "intent": intent, "v2_hit": old_hit,
                                "v3_hit": new_hit})
    output = {"scope": "untouched 98 holdout packages, simulated A view",
              "comparison": "V2 vs V3 top5 exact intact nonempty gold chain",
              "gold_opened_after_100_model_attempts": True,
              "groups": {kind: dict(count) for kind, count in sorted(groups.items())},
              "changed": changed}
    path = OUT_DIR / "a_candidate_v3_holdout.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({"path": str(path), "groups": output["groups"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
