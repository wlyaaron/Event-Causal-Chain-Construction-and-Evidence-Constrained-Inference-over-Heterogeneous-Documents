"""Census annotation choice for the recurring direct-effect question form.

Descriptive only. The fit split is shown separately so the annotation pattern
can be studied without relying on held-out packages to formulate a rule.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "数据集" / "训练集"
OUT = ROOT / "experiments" / "research_audit_20261010" / "direct_effect_gold_priority.json"
IDS = re.compile(r"D\d{3,}")
PHRASE = "之后直接发生了什么"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    splits = read(ROOT / "experiments" / "finetune_split_20261005.json")["splits"]
    split_of = {pack: name for name, packs in splits.items() for pack in packs}
    counts = collections.defaultdict(collections.Counter)
    exceptions = []
    for pack in sorted(TRAIN.iterdir()):
        if not pack.is_dir() or pack.name not in split_of:
            continue
        questions = read(pack / "问题.json")
        gold = {x["sample_id"]: x for x in read(pack / "gold" / "问答对_答案.json")}
        edges = read(pack / "事件因果关系列表.json")
        for question in questions:
            if PHRASE not in question["question"]:
                continue
            sid = question["sample_id"]
            mentioned = IDS.findall(question["question"])
            if not mentioned:
                raise ValueError(f"No event in question: {sid}")
            start = mentioned[0]
            direct = sorted({edge.get("result_event_id") or edge.get("effect_event_id")
                             for edge in edges if edge.get("cause_event_id") == start
                             and (edge.get("causal_type") or edge.get("relation_type")) == "直接因果"})
            if not direct:
                raise ValueError(f"No direct successor: {sid}")
            chains = gold[sid].get("evidence_chains") or []
            split = split_of[pack.name]
            bucket = counts[split]
            bucket["questions"] += 1
            bucket["multiple_direct_successors"] += len(direct) > 1
            bucket["gold_is_first_direct_successor"] += chains == [[start, direct[0]]]
            bucket["gold_D001_D002"] += chains == [["D001", "D002"]]
            if chains != [[start, direct[0]]]:
                exceptions.append({"sample_id": sid, "pack": pack.name,
                                   "split": split, "start": start,
                                   "direct_successors": direct, "gold_chains": chains,
                                   "question": question["question"]})
    result = {"scope": "organizer training packages, phrase containing '之后直接发生了什么'",
              "note": "The first direct successor is the smallest event ID; this is an annotation regularity, not proof that it is semantically superior.",
              "by_split": {split: dict(counts[split]) for split in splits},
              "all": dict(sum(counts.values(), collections.Counter())),
              "exceptions": exceptions}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all": result["all"], "by_split": result["by_split"],
                      "exceptions": len(exceptions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
