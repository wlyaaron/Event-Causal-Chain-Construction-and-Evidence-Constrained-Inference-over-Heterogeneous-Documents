"""Freeze a pack-disjoint training validation slice for paired API research."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import task4_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=4)
    parser.add_argument("--ranker-validation-only", action="store_true",
                        help="Select from the event ranker's 200-pack validation split")
    args = parser.parse_args()
    samples = core.discover_samples(core.REPO_ROOT / "数据集" / "训练集",
                                    limit=0)
    packs = sorted({sample.pack for sample in samples})
    exploratory_packs = set(random.Random(20261004).sample(packs, 40))
    old_predictions = core.REPO_ROOT / "outputs" / "deepseek_flash_train_100_research.json"
    old_ids = {row["sample_id"] for row in core.read_json(old_predictions)}
    old_packs = {sample.pack for sample in samples if sample.sample_id in old_ids}
    ranker_validation = set(packs)
    if args.ranker_validation_only:
        shuffled = list(packs)
        random.Random(20261004).shuffle(shuffled)
        ranker_validation = set(shuffled[800:])
    eligible = [sample for sample in samples
                if sample.pack not in exploratory_packs | old_packs
                and sample.pack in ranker_validation]
    rng = random.Random(20261005)
    rng.shuffle(eligible)
    chosen = []
    used_packs = set()
    counts = {kind: 0 for kind in ("retrospective", "prospective",
                                  "counterfactual", "unanswerable")}
    for sample in eligible:
        if (sample.question_type not in counts or
                counts[sample.question_type] >= args.per_type or
                sample.pack in used_packs):
            continue
        chosen.append(sample)
        used_packs.add(sample.pack)
        counts[sample.question_type] += 1
        if all(count == args.per_type for count in counts.values()):
            break
    if any(count != args.per_type for count in counts.values()):
        raise ValueError(f"Could not fill all held-out types: {counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([sample.sample_id for sample in chosen],
                                      ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    manifest = [{"sample_id": sample.sample_id,
                 "question_type": sample.question_type,
                 "pack": sample.pack.name} for sample in chosen]
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"count": len(chosen), "by_type": counts,
                      "excluded_exploration_packs": len(exploratory_packs | old_packs),
                      "ranker_validation_only": args.ranker_validation_only},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
