"""Preserve every organizer TRAIN gold item verbatim in an ignored sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import task4_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("original gold sidecar must stay under ignored outputs/")
    seen = set()
    chains = 0
    multi = 0
    packs = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for pack in sorted(args.train_root.iterdir()):
            gold_path = pack / "gold" / "问答对_答案.json"
            if not gold_path.is_file():
                continue
            packs += 1
            for item in core.read_json(gold_path):
                key = (pack.name, item["sample_id"])
                if key in seen:
                    raise ValueError(f"duplicate organizer gold key: {key}")
                seen.add(key)
                alternatives = item.get("evidence_chains") or []
                chains += len(alternatives)
                multi += len(alternatives) > 1
                output.write(json.dumps({"pack": pack.name, "gold": item},
                                        ensure_ascii=False) + "\n")
    digest = hashlib.sha256()
    with args.output.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    print(json.dumps({"packs": packs, "questions": len(seen),
                      "gold_chains": chains, "questions_with_multiple_chains": multi,
                      "sha256": digest.hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
