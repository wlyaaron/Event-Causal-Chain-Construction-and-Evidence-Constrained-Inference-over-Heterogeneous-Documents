"""Freeze 24 gold-stratified fit questions before refusal-format model calls.

Gold determines only the 8/8/8 sampling strata. The manifest and API input
contain no gold answer, chain, or required fact.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core
import task4_workflow as workflow


TRAIN = ROOT / "数据集/训练集"
OUT = ROOT / "experiments/refusal_format24_20261010.json"
SPLIT = ROOT / "experiments/finetune_split_20261005.json"
PRIOR = (
    ROOT / "experiments/reasoning_type_90_20261009.json",
    ROOT / "experiments/reasoning_type_fresh_30_20261009.json",
    ROOT / "experiments/icl_diagnostic_12_20261009.json",
)
PROMPT_V4 = ROOT / "experiments/prompts/task4_merged_rules_v4_draft_20261009.txt"
PROMPT_FORMAT = ROOT / "experiments/prompts/task4_refusal_format_v7_20261010.txt"
SEED = "task4:refusal-format24:2026-10-10:before-api"
VIEWS = {"strict": "AAAAABBC", "explanatory": "AAAAABBC", "other": "AAAABBBC"}


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str)
                           else value).hexdigest()


def doc_hashes(pack: Path) -> set[str]:
    return {digest(re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).strip())
            for path in pack.glob("D[0-9]*.txt")}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Manifest already frozen: {OUT}")
    fit = set(core.read_json(SPLIT)["splits"]["fit"])
    excluded = {row["pack"] for path in PRIOR for row in core.read_json(path)["rows"]}
    used_packs = set(excluded)
    used_docs = set().union(*(doc_hashes(TRAIN / name) for name in excluded))
    candidates: dict[str, list[tuple[str, dict]]] = {key: [] for key in VIEWS}
    for pack_name in sorted(fit - excluded):
        pack = TRAIN / pack_name
        gold = {item["sample_id"]: item for item in
                core.read_json(pack / "gold/问答对_答案.json")}
        for question in core.read_json(pack / "问题.json"):
            answer = str(gold[question["sample_id"]].get("answers", "")).strip()
            group = ("strict" if answer == "无法确定" else
                     "explanatory" if question["question_type"] == "unanswerable"
                     else "other")
            candidates[group].append((pack_name, question))
    rows = []
    for group, views in VIEWS.items():
        ranked = sorted(candidates[group], key=lambda item:
                        digest(f"{SEED}:{group}:{item[0]}:{item[1]['sample_id']}"))
        chosen = []
        for pack_name, question in ranked:
            if pack_name in used_packs:
                continue
            pack = TRAIN / pack_name
            hashes = doc_hashes(pack)
            if hashes & used_docs:
                continue
            chosen.append((pack_name, question))
            used_packs.add(pack_name)
            used_docs.update(hashes)
            if len(chosen) == len(views):
                break
        if len(chosen) != len(views):
            raise RuntimeError(f"Only {len(chosen)} isolated {group} questions")
        for (pack_name, question), view in zip(chosen, views):
            sample = core.Sample(TRAIN / pack_name, "train", question["sample_id"],
                                 question["question_type"], question["question"])
            content, _, _ = workflow.training_view_input(sample, view)
            rows.append({"sample_id": sample.sample_id, "pack": pack_name,
                         "reasoning_type": question.get("reasoning_type"),
                         "question_type": sample.question_type,
                         "question": sample.question, "view": view,
                         "input_chars": len(content),
                         "input_sha256": digest(content)})
    if len(rows) != 24 or len({row["pack"] for row in rows}) != 24:
        raise AssertionError("Unexpected sample count or duplicate pack")
    payload = {"purpose": "Paired V4 vs short refusal-format prompt; 8 strict, 8 explanatory, 8 other gold strata",
               "seed": SEED, "source_split": "fit", "n": len(rows),
               "gold_used_only_for_sampling": True,
               "excluded_prior_pack_count": len(excluded),
               "document_isolated_from_prior_and_within_set": True,
               "views": {view: sum(row["view"] == view for row in rows)
                         for view in "ABC"},
               "prompt_sha256": {"v4": digest(PROMPT_V4.read_bytes()),
                                 "fmt": digest(PROMPT_FORMAT.read_bytes())},
               "rows": rows}
    core.atomic_json(OUT, payload)
    print(json.dumps({"manifest": str(OUT), "n": len(rows),
                      "views": payload["views"],
                      "max_input_chars": max(row["input_chars"] for row in rows)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
