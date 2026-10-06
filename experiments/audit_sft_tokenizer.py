"""Audit complete SFT sequences and gold evidence positions with a real tokenizer.

Reads organizer TRAIN views only. This reports truncation risk; it never truncates
or writes examples, and gold chains are used solely to locate source documents.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from experiments.audit_training_data import quantiles


def inspect_row(row: dict, tokenizer, limits: list[int]) -> dict:
    messages = row["messages"]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ends = [end for _, end in encoded["offset_mapping"]]
    length = len(encoded["input_ids"])
    user = json.loads(messages[1]["content"])
    documents = {doc["doc_id"]: doc for doc in user["documents"]}
    events = {event["event_id"]: event for event in user.get("events") or []}
    gold_ids = {node for chain in row["compatible_gold_chains"] for node in chain}
    evidence_end_tokens = []
    unresolved = 0
    event_only = 0
    for node in gold_ids:
        event = events.get(node)
        doc_id = event.get("doc_id") if event else node
        doc = documents.get(doc_id)
        evidence = doc if doc is not None else event
        if evidence is None:
            unresolved += 1
            continue
        event_only += doc is None
        needle = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        start = rendered.find(needle)
        if start < 0:
            unresolved += 1
            continue
        evidence_end_tokens.append(bisect.bisect_left(ends, start + len(needle)) + 1)
    return {"tokens": length, "evidence_end_tokens": evidence_end_tokens,
            "unresolved_evidence_ids": unresolved,
            "event_only_evidence_ids": event_only,
            "over": {str(limit): length > limit for limit in limits},
            "lost_evidence": {str(limit): sum(pos > limit for pos in evidence_end_tokens)
                              for limit in limits}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--limits", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                              use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer with offset mappings is required")
    limits = sorted(set(args.limits))
    groups = defaultdict(lambda: {"lengths": [], "positions": [], "over": Counter(),
                                  "lost_rows": Counter(), "lost_ids": Counter(),
                                  "unresolved_ids": 0, "event_only_ids": 0})
    for split in ("fit", "validation", "holdout"):
        for view in "ABC":
            path = args.data_dir / f"{split}_{view}.jsonl"
            group = groups[f"{split}_{view}"]
            with path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    info = inspect_row(row, tokenizer, limits)
                    group["lengths"].append(info["tokens"])
                    group["unresolved_ids"] += info["unresolved_evidence_ids"]
                    group["event_only_ids"] += info["event_only_evidence_ids"]
                    group["positions"].extend(round(100 * pos / info["tokens"])
                                              for pos in info["evidence_end_tokens"])
                    for limit in limits:
                        key = str(limit)
                        group["over"][key] += info["over"][key]
                        group["lost_rows"][key] += info["lost_evidence"][key] > 0
                        group["lost_ids"][key] += info["lost_evidence"][key]
    report = {"model_revision": args.model_revision,
              "tokenizer": str(args.tokenizer), "limits": limits,
              "groups": {name: {"rows": len(group["lengths"]),
                                "token_quantiles": quantiles(group["lengths"]),
                                "gold_evidence_end_position_percent_quantiles":
                                    quantiles(group["positions"]),
                                "unresolved_gold_evidence_ids": group["unresolved_ids"],
                                "event_without_source_document_ids": group["event_only_ids"],
                                "rows_exceeding_limit": dict(group["over"]),
                                "rows_with_gold_evidence_beyond_limit":
                                    dict(group["lost_rows"]),
                                "gold_evidence_ids_beyond_limit": dict(group["lost_ids"])}
                         for name, group in sorted(groups.items())},
              "note": "Gold evidence positions use the source document when present, otherwise the visible event record. A selected chain may use fewer IDs than the union of alternatives; no prompt was truncated."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({name: {"rows": group["rows"],
                             "token_quantiles": group["token_quantiles"],
                             "rows_exceeding_limit": group["rows_exceeding_limit"],
                             "rows_with_gold_evidence_beyond_limit":
                                 group["rows_with_gold_evidence_beyond_limit"],
                             "event_without_source_document_ids":
                                 group["event_without_source_document_ids"]}
                      for name, group in report["groups"].items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
