"""Audit gold for the recurring precise-quantification question wording."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402


def main() -> None:
    counts: Counter[str] = Counter()
    exceptions = []
    for question_file in sorted((ROOT / "数据集/训练集").glob("*/问题.json")):
        matches = [q for q in core.read_json(question_file)
                   if "能否精确量化" in q["question"]]
        if not matches:
            continue
        gold_path = question_file.parent / "gold/问答对_答案.json"
        gold = {g["sample_id"]: g for g in core.read_json(gold_path)}
        for question in matches:
            answer = str(gold[question["sample_id"]].get("answers", "")).strip()
            strict = answer == "无法确定"
            counts["all"] += 1
            counts["strict" if strict else "explanatory"] += 1
            if not strict:
                exceptions.append({"sample_id": question["sample_id"],
                                   "question": question["question"],
                                   "gold_answer": answer,
                                   "gold_chains": gold[question["sample_id"]]
                                   .get("evidence_chains")})
    report = {"counts": dict(counts), "exceptions": exceptions}
    out = ROOT / "experiments/results/task4_refusal_quantity_audit_20261010.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"{out}: {dict(counts)}; exceptions={len(exceptions)}")


if __name__ == "__main__":
    main()
