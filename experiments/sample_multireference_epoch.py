"""Make a deterministic TRAIN-only SFT epoch from intact gold alternatives.

Input and output remain under ignored outputs/. Validation and holdout rows are
left unchanged; this script never reads a hosted-model response or test answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import task4_core as core


def selected_chain(row: dict, seed: int, epoch: int) -> list[str]:
    alternatives = row["compatible_gold_chains"]
    if not alternatives or any(not isinstance(chain, list) for chain in alternatives):
        raise ValueError("row has no compatible complete gold alternatives")
    key = f"{seed}:{epoch}:{row['pack']}:{row['sample_id']}:{row['view']}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return list(alternatives[int.from_bytes(digest[:8], "big") % len(alternatives)])


def build_epoch(source: Path, destination: Path, seed: int, epoch: int) -> int:
    ignored = (core.REPO_ROOT / "outputs").resolve()
    if not source.resolve().is_relative_to(ignored) or not destination.resolve().is_relative_to(ignored):
        raise ValueError("training data and epoch outputs must stay under ignored outputs/")
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            target = json.loads(row["messages"][-1]["content"])
            target["evidence_chain"] = selected_chain(row, seed, epoch)
            if (target["answer"].strip() == "无法确定") != (target["evidence_chain"] == []):
                raise ValueError(f"answer/chain refusal conflict: {row['pack']} {row['sample_id']}")
            row["messages"][-1]["content"] = json.dumps(target, ensure_ascii=False)
            row["sampled_gold_epoch"] = epoch
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps({"examples": build_epoch(args.source, args.output,
                                                args.seed, args.epoch),
                      "seed": args.seed, "epoch": args.epoch}, ensure_ascii=False))


if __name__ == "__main__":
    main()
