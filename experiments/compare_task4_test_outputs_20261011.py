"""Compare two complete blind-test prediction files without reading gold."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import task4_core as core


def summarize(rows: list[dict], edges_by_id: dict[str, set[tuple[str, str]]]) -> dict:
    n = len(rows)
    left_lengths = [len(row["left"]["evidence_chain"]) for row in rows]
    right_lengths = [len(row["right"]["evidence_chain"]) for row in rows]
    edge_counts = Counter()
    for row in rows:
        edges = edges_by_id[row["sample_id"]]
        if not edges:
            continue
        for side in ("left", "right"):
            chain = row[side]["evidence_chain"]
            for pair in zip(chain, chain[1:]):
                edge_counts[f"{side}_pairs"] += 1
                edge_counts[f"{side}_supported"] += pair in edges
    return {
        "questions": n,
        "answer_exact_same": sum(row["answer_same"] for row in rows),
        "chain_exact_same": sum(row["chain_same"] for row in rows),
        "chain_same_answer_changed": sum(row["chain_same"] and not row["answer_same"] for row in rows),
        "both_changed": sum(not row["chain_same"] and not row["answer_same"] for row in rows),
        "answer_similarity_mean": round(sum(row["answer_similarity"] for row in rows) / n, 4),
        "left_answer_chars_mean": round(sum(len(row["left"]["answer"]) for row in rows) / n, 2),
        "right_answer_chars_mean": round(sum(len(row["right"]["answer"]) for row in rows) / n, 2),
        "left_chain_nodes_mean": round(sum(left_lengths) / n, 3),
        "right_chain_nodes_mean": round(sum(right_lengths) / n, 3),
        "right_chain_shorter": sum(r < l for l, r in zip(left_lengths, right_lengths)),
        "right_chain_longer": sum(r > l for l, r in zip(left_lengths, right_lengths)),
        "left_strict_refusals": sum(row["left"]["answer"] == "无法确定" for row in rows),
        "right_strict_refusals": sum(row["right"]["answer"] == "无法确定" for row in rows),
        "answer_to_strict_refusal": sum(row["left"]["answer"] != "无法确定"
                                        and row["right"]["answer"] == "无法确定" for row in rows),
        "strict_refusal_to_answer": sum(row["left"]["answer"] == "无法确定"
                                        and row["right"]["answer"] != "无法确定" for row in rows),
        "left_graph_pairs_supported": dict(pairs=edge_counts["left_pairs"],
                                            supported=edge_counts["left_supported"]),
        "right_graph_pairs_supported": dict(pairs=edge_counts["right_pairs"],
                                             supported=edge_counts["right_supported"]),
    }


def compare(left_path: Path, right_path: Path) -> dict:
    samples = core.discover_samples(core.DEFAULT_DATASET, "all", 0)
    core.validate_file(left_path, samples, 80000)
    core.validate_file(right_path, samples, 80000)
    left = {row["sample_id"]: row for row in core.read_json(left_path)}
    right = {row["sample_id"]: row for row in core.read_json(right_path)}
    edges_by_id = {sample.sample_id: core.build_input(sample)[2] for sample in samples}
    grouped = defaultdict(list)
    all_rows = []
    for sample in samples:
        a, b = left[sample.sample_id], right[sample.sample_id]
        item = {
            "sample_id": sample.sample_id, "track": sample.track,
            "question_type": sample.question_type,
            "answer_same": a["answer"] == b["answer"],
            "chain_same": a["evidence_chain"] == b["evidence_chain"],
            "answer_similarity": round(SequenceMatcher(None, a["answer"], b["answer"]).ratio(), 4),
            "left": {key: a[key] for key in ("answer", "evidence_chain", "confidence")},
            "right": {key: b[key] for key in ("answer", "evidence_chain", "confidence")},
        }
        grouped[sample.track].append(item)
        all_rows.append(item)
    return {"left": str(left_path.resolve()), "right": str(right_path.resolve()),
            "scope": "test output behavior only; no gold and no correctness inference",
            "summary": {track: summarize(grouped[track], edges_by_id)
                        for track in ("A", "B", "C")},
            "overall": summarize(all_rows, edges_by_id),
            "rows": all_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("Per-question comparisons must stay in ignored outputs/")
    result = compare(args.left, args.right)
    core.atomic_json(args.output, result)
    print(json.dumps({"summary": result["summary"], "overall": result["overall"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
