"""Build deterministic A/B/C SFT mixes from organizer TRAIN views only."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
from collections import Counter
from pathlib import Path

import task4_core as core
from experiments.sample_multireference_epoch import selected_chain


def _fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def mixed_rows(source: dict[str, list[dict]], *, epochs: int, seed: int,
               multireference: bool, weights: dict[str, float],
               exclude_flags: set[str], a_graph_supported_only: bool,
               start_epoch: int = 0):
    if epochs < 1 or any(weights.get(view, 0) < 0 for view in "ABC"):
        raise ValueError("epochs and view weights must be valid")
    by_key = {}
    for view in "ABC":
        for row in source.get(view, []):
            if row["view"] != view:
                raise ValueError("view/file mismatch")
            key = (row["pack"], row["sample_id"])
            if view in by_key.setdefault(key, {}):
                raise ValueError(f"duplicate training view: {key} {view}")
            by_key[key][view] = row
    for epoch in range(start_epoch, start_epoch + epochs):
        for key, alternatives in sorted(by_key.items()):
            available = [view for view in "ABC" if view in alternatives
                         and not (set(alternatives[view]["flags"]) & exclude_flags)
                         and not (view == "A" and a_graph_supported_only
                                  and alternatives[view]["visible_graph_supported_chains"] == 0)
                         and weights.get(view, 0) > 0]
            total = sum(weights[view] for view in available)
            if total == 0:
                continue
            draw = _fraction(f"{seed}:{epoch}:{key[0]}:{key[1]}") * total
            chosen_view = available[-1]
            for view in available:
                draw -= weights[view]
                if draw < 0:
                    chosen_view = view
                    break
            row = dict(alternatives[chosen_view])
            row["messages"] = [dict(message) for message in row["messages"]]
            if multireference:
                target = json.loads(row["messages"][-1]["content"])
                target["evidence_chain"] = selected_chain(row, seed, epoch)
                row["messages"][-1]["content"] = json.dumps(target, ensure_ascii=False)
            row["mixed_epoch"] = epoch
            row["reference_mode"] = "sampled" if multireference else "fixed"
            yield row


def _key(row: dict) -> tuple[str, str]:
    return row["pack"], row["sample_id"]


def _sorted_rows(path: Path, view: str):
    previous = None
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            key = _key(row)
            if row["view"] != view or (previous is not None and key <= previous):
                raise ValueError(f"unsorted or duplicated {view} source: {path}")
            previous = key
            yield row


def stream_mixed_rows(data_dir: Path, *, epochs: int, seed: int,
                      multireference: bool, weights: dict[str, float],
                      exclude_flags: set[str], a_graph_supported_only: bool):
    """Merge the three sorted fit views without loading them into RAM."""
    for epoch in range(epochs):
        sources = [_sorted_rows(data_dir / f"fit_{view}.jsonl", view)
                   for view in "ABC"]
        for _, grouped in itertools.groupby(heapq.merge(*sources, key=_key), key=_key):
            alternatives = {row["view"]: [row] for row in grouped}
            yield from mixed_rows(alternatives, epochs=1, seed=seed,
                                  multireference=multireference, weights=weights,
                                  exclude_flags=exclude_flags,
                                  a_graph_supported_only=a_graph_supported_only,
                                  start_epoch=epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--weights", type=float, nargs=3, default=[.6, .3, .1])
    parser.add_argument("--multireference", action="store_true")
    parser.add_argument("--a-graph-supported-only", action="store_true")
    parser.add_argument("--exclude-flag", action="append", default=[])
    args = parser.parse_args()
    ignored = (core.REPO_ROOT / "outputs").resolve()
    if not args.data_dir.resolve().is_relative_to(ignored) or not args.output.resolve().is_relative_to(ignored):
        raise ValueError("SFT input and output must stay in ignored outputs/")
    weights = dict(zip("ABC", args.weights))
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for row in stream_mixed_rows(args.data_dir, epochs=args.epochs,
                                     seed=args.seed,
                                     multireference=args.multireference,
                                     weights=weights,
                                     exclude_flags=set(args.exclude_flag),
                                     a_graph_supported_only=args.a_graph_supported_only):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[row["view"]] += 1
    digest = hashlib.sha256()
    with args.output.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    print(json.dumps({"rows": sum(counts.values()), "by_view": dict(counts),
                      "epochs": args.epochs, "multireference": args.multireference,
                      "sha256": digest.hexdigest()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
