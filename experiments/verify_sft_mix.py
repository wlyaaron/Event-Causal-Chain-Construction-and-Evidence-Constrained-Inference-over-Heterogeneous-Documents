"""Verify mixed TRAIN examples contain only one intact allowed gold chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import task4_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=core.REPO_ROOT / "experiments" / "finetune_split_20261005.json")
    args = parser.parse_args()
    if not args.source.resolve().is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("mixed training data must stay under ignored outputs/")
    fit_packs = set(core.read_json(args.manifest)["splits"]["fit"])
    counts = Counter()
    seen = set()
    with args.source.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            key = (row["mixed_epoch"], row["pack"], row["sample_id"])
            if key in seen:
                raise ValueError(f"duplicate question in epoch: {key}")
            seen.add(key)
            if row["pack"] not in fit_packs:
                raise ValueError(f"non-fit pack: {row['pack']}")
            target = json.loads(row["messages"][-1]["content"])
            if set(target) != {"answer", "evidence_chain"}:
                raise ValueError("unexpected target fields")
            if target["evidence_chain"] not in row["compatible_gold_chains"]:
                raise ValueError(f"assembled gold chain: {key}")
            if (target["answer"].strip() == "无法确定") != (target["evidence_chain"] == []):
                raise ValueError(f"refusal conflict: {key}")
            counts[f"epoch_{row['mixed_epoch']}_{row['view']}"] += 1
    digest = hashlib.sha256()
    with args.source.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    report = {"rows": sum(counts.values()), "by_epoch_view": dict(sorted(counts.items())),
              "sha256": digest.hexdigest()}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
