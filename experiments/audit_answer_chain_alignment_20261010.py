"""Descriptive census of answer ID references versus intact gold alternatives.

Explicit IDs are a necessary check, not a semantic proof that every answer
fact is supported by the chain. No model output is read here.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "数据集" / "训练集"
OUT = ROOT / "experiments" / "research_audit_20261010"
ID = re.compile(r"(?<![A-Za-z0-9])D\d{3,}(?!\d)")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    split = read(ROOT / "experiments" / "finetune_split_20261005.json")["splits"]
    split_of = {pack: name for name, packs in split.items() for pack in packs}
    counts = collections.Counter()
    by_split = collections.defaultdict(collections.Counter)
    by_reasoning = collections.defaultdict(collections.Counter)
    exceptions = []
    no_ids = []
    for pack in sorted(TRAIN.iterdir()):
        if not pack.is_dir() or pack.name not in split_of:
            continue
        gold = read(pack / "gold" / "问答对_答案.json")
        events = {item["event_id"] for item in read(pack / "事件列表.json")}
        for item in gold:
            answer = item.get("answers") or ""
            chains = item.get("evidence_chains") or []
            raw_refs = set(ID.findall(answer))
            refs = raw_refs & events
            valid = [set(chain) for chain in chains if chain]
            facts = item.get("answer_facts") or {}
            required_refs = set(ID.findall(" ".join(facts.get("required_facts") or [])))
            record = {
                "pack": pack.name, "split": split_of[pack.name],
                "sample_id": item["sample_id"], "question_type": item.get("question_type"),
                "reasoning_type": item.get("reasoning_type"), "answer": answer,
                "answer_ids": sorted(refs), "required_fact_ids": sorted(required_refs),
                "gold_chains": chains, "id_like_text_not_event": sorted(raw_refs - events),
            }
            flags = ["questions"]
            if chains:
                flags.append("with_chain")
            else:
                flags.append("without_chain")
            if refs:
                flags.append("answer_with_id")
                if valid and any(refs <= chain for chain in valid):
                    flags.append("answer_ids_in_one_complete_chain")
                elif valid:
                    flags.append("answer_ids_outside_every_complete_chain")
                    record["out_of_chain_ids"] = sorted(refs - set.union(*valid))
                    exceptions.append(record)
                else:
                    flags.append("answer_ids_but_no_chain")
                    exceptions.append(record)
            else:
                flags.append("answer_without_id")
                if len(no_ids) < 30:
                    no_ids.append(record)
            if required_refs and valid and not any(required_refs <= chain for chain in valid):
                flags.append("required_fact_ids_outside_every_complete_chain")
            for flag in flags:
                counts[flag] += 1
                by_split[split_of[pack.name]][flag] += 1
                by_reasoning[item.get("reasoning_type") or "missing"][flag] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"note": "ID reference inclusion is not semantic support; compare each intact alternative, never splice gold chains.",
              "counts": dict(counts), "by_split": {k: dict(v) for k, v in by_split.items()},
              "by_reasoning": {k: dict(v) for k, v in by_reasoning.items()},
              "exceptions": exceptions, "no_id_examples": no_ids}
    (OUT / "answer_chain_alignment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "exceptions": len(exceptions), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
