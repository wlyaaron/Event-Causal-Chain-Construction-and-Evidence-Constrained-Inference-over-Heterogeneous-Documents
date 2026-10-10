"""Regroup the completed fit candidate audit by an input-only applicability rule.

This is exploratory scope selection on fit, not an untouched validation result.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "数据集" / "训练集"
SPLIT = ROOT / "experiments" / "finetune_split_20261005.json"
AUDIT = ROOT / "experiments" / "research_audit_20261010" / "a_candidate_pool_fit.json"
OUT = ROOT / "experiments" / "research_audit_20261010" / "a_candidate_scope_fit.json"

# Uses only the question text and the provided question_type. No gold fields.
PATH_CUE = re.compile(
    r"因果链|证据链|有序|一步步|多跳|传导|依次|追溯|事件链|因果顺序|"
    r"如何.*发生|怎么.*发生|形成过程"
)


def main() -> None:
    fit_packs = json.loads(SPLIT.read_text(encoding="utf-8"))["splits"]["fit"]
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in audit["rows"]}
    groups: dict[str, Counter[str]] = defaultdict(Counter)

    for pack_name in fit_packs:
        questions = json.loads((TRAIN / pack_name / "问题.json").read_text(encoding="utf-8"))
        for question in questions:
            sample_id = question["sample_id"]
            question_type = question["question_type"]
            group_names = [question_type]
            if question_type == "retrospective" and PATH_CUE.search(question["question"]):
                group_names.append("known_retrospective_path_cue")
            for group_name in group_names:
                count = groups[group_name]
                count["all"] += 1
                row = by_id.get(sample_id)
                if row is None:
                    count["empty_gold_chain"] += 1
                    continue
                count["nonempty_gold_chain"] += 1
                count["multi_any_gold_chain"] += any(
                    length > 1 for length in row["gold_chain_lengths"]
                )
                count["raw_graph_hit"] += row["raw_graph_hit"]
                count["old_top5_hit"] += row["old_top5_hit"]
                for k in (5, 10, 16):
                    count[f"new_top{k}_hit"] += row["new_top_k_hit"][str(k)]

    output = {
        "scope": "fit only, simulated A view",
        "known_path_rule": "question_type=retrospective AND PATH_CUE matches question text",
        "path_cue_pattern": PATH_CUE.pattern,
        "caution": "Applicability rule chosen after fit exploration; validation is required before generalization.",
        "groups": {name: dict(sorted(count.items())) for name, count in sorted(groups.items())},
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: output["groups"].get(name) for name in
                      ("known_retrospective_path_cue", "retrospective", "prospective",
                       "counterfactual", "unanswerable")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
