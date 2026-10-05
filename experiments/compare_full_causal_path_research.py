"""Compare two blind-test research outputs without claiming an accuracy score.

The test set has no gold. This reports output stability and graph consistency,
not answer correctness or the contest's hidden score. Report stays in outputs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import task4_core as core


def compare(old_path: Path, new_path: Path, dataset: Path) -> dict:
    samples = core.discover_samples(dataset, "all", 0)
    core.validate_file(old_path, samples, 80000)
    core.validate_file(new_path, samples, 80000)
    old = {row["sample_id"]: row for row in core.read_json(old_path)}
    new = {row["sample_id"]: row for row in core.read_json(new_path)}
    groups = defaultdict(list)
    changes = []
    for sample in samples:
        before, after = old[sample.sample_id], new[sample.sample_id]
        _, _, given_edges = core.build_input(sample)
        old_chain, new_chain = before["evidence_chain"], after["evidence_chain"]
        old_steps = list(zip(old_chain, old_chain[1:]))
        new_steps = list(zip(new_chain, new_chain[1:]))
        row = {"sample_id": sample.sample_id, "track": sample.track,
               "question_type": sample.question_type,
               "answer_same": before["answer"] == after["answer"],
               "chain_same": old_chain == new_chain,
               "old_refusal": before["answer"] == "无法确定",
               "new_refusal": after["answer"] == "无法确定",
               "old_answer_chars": len(before["answer"]),
               "new_answer_chars": len(after["answer"]),
               "old_chain_nodes": len(old_chain), "new_chain_nodes": len(new_chain),
               "old_given_edge_steps": sum(edge in given_edges for edge in old_steps),
               "new_given_edge_steps": sum(edge in given_edges for edge in new_steps),
               "old_chain_steps": len(old_steps), "new_chain_steps": len(new_steps)}
        groups[sample.track].append(row)
        groups["all"].append(row)
        groups[f"{sample.track}/{sample.question_type}"].append(row)
        if not row["answer_same"] or not row["chain_same"]:
            changes.append({"sample_id": sample.sample_id, "track": sample.track,
                            "question_type": sample.question_type,
                            "old_chain": old_chain, "new_chain": new_chain,
                            "old_answer_chars": row["old_answer_chars"],
                            "new_answer_chars": row["new_answer_chars"],
                            "old_refusal": row["old_refusal"],
                            "new_refusal": row["new_refusal"]})

    def summary(rows: list[dict], name: str) -> dict:
        n = len(rows)
        result = {"questions": n,
                  "answer_same": sum(row["answer_same"] for row in rows),
                  "chain_same": sum(row["chain_same"] for row in rows),
                  "both_same": sum(row["answer_same"] and row["chain_same"] for row in rows),
                  "old_refusals": sum(row["old_refusal"] for row in rows),
                  "new_refusals": sum(row["new_refusal"] for row in rows),
                  "refusal_to_answer": sum(row["old_refusal"] and not row["new_refusal"] for row in rows),
                  "answer_to_refusal": sum(not row["old_refusal"] and row["new_refusal"] for row in rows),
                  "chain_shorter": sum(row["new_chain_nodes"] < row["old_chain_nodes"] for row in rows),
                  "chain_longer": sum(row["new_chain_nodes"] > row["old_chain_nodes"] for row in rows),
                  "old_mean_answer_chars": round(sum(row["old_answer_chars"] for row in rows) / n, 2),
                  "new_mean_answer_chars": round(sum(row["new_answer_chars"] for row in rows) / n, 2),
                  "old_mean_chain_nodes": round(sum(row["old_chain_nodes"] for row in rows) / n, 3),
                  "new_mean_chain_nodes": round(sum(row["new_chain_nodes"] for row in rows) / n, 3)}
        if name == "A" or name.startswith("A/"):
            old_steps = sum(row["old_chain_steps"] for row in rows)
            new_steps = sum(row["new_chain_steps"] for row in rows)
            result["old_given_edge_step_fraction"] = round(
                sum(row["old_given_edge_steps"] for row in rows) / old_steps, 4) if old_steps else None
            result["new_given_edge_step_fraction"] = round(
                sum(row["new_given_edge_steps"] for row in rows) / new_steps, 4) if new_steps else None
        return result

    progress = new_path.with_name(new_path.stem + ".progress.json")
    usage = Counter()
    operations = {}
    if progress.exists():
        progress_rows = core.read_json(progress)["rows"]
        for item in progress_rows:
            usage.update({key: value for key, value in item["usage"].items()
                          if isinstance(value, int) and not isinstance(value, bool)})
        operations = {"A_format_fallbacks": sum(bool(item["usage"].get("format_fallback"))
                                                 for item in progress_rows),
                      "input_chars": sum(item["input_chars"] for item in progress_rows)}
    return {"note": "Blind-test output comparison only; no gold, no accuracy or official score.",
            "old_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
            "new_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest(),
            "reported_usage_lower_bound": dict(usage),
            "usage_note": "Only A reports usage; B/C use the original baseline runner without token accounting.",
            "operations": operations,
            "by_group": {name: summary(rows, name) for name, rows in groups.items()},
            "changed": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=core.DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("research comparison must stay in ignored outputs/")
    report = compare(args.old, args.new, args.dataset)
    core.atomic_json(args.output, report)
    print(json.dumps({name: report["by_group"][name] for name in ("A", "B", "C", "all")
                      if name in report["by_group"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
