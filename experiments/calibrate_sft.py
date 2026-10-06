"""Fit numeric confidence only on package-isolated validation predictions.

Confidence means probability of a declared local joint-success proxy, not an
official score. Refusals retain JSON null confidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import task4_core as core
from experiments.evaluate_sft import score_row


def _length_bucket(chain: list[str]) -> str:
    return str(len(chain)) if len(chain) <= 2 else "3+"


def fit_rates(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        if row.get("answer") == "无法确定" or not row.get("evidence_chain"):
            continue
        groups[row["view"]].append(bool(row["success"]))
        groups[f"{row['view']}:{_length_bucket(row['evidence_chain'])}"].append(
            bool(row["success"]))
    return {key: {"n": len(values), "successes": sum(values),
                  "rate": (sum(values) + 2) / (len(values) + 4)}
            for key, values in groups.items()}


def calibrated_confidence(prediction: dict, rates: dict) -> float | None:
    if prediction["answer"] == "无法确定":
        return None
    view = prediction["view"]
    specific = rates.get(f"{view}:{_length_bucket(prediction['evidence_chain'])}")
    chosen = specific if specific and specific["n"] >= 20 else rates.get(view)
    if chosen is None:
        raise ValueError(f"no validation calibration for view {view}")
    return round(chosen["rate"], 4)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--validation-predictions", type=Path, required=True)
    fit.add_argument("--data-dir", type=Path, required=True)
    fit.add_argument("--train-root", type=Path,
                     default=core.REPO_ROOT / "数据集" / "训练集")
    fit.add_argument("--output", type=Path, required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("--predictions", type=Path, required=True)
    apply.add_argument("--calibration", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ignored = (core.REPO_ROOT / "outputs").resolve()
    paths = ([args.validation_predictions, args.data_dir, args.output]
             if args.mode == "fit" else [args.predictions, args.calibration, args.output])
    if any(not path.resolve().is_relative_to(ignored) for path in paths):
        raise ValueError("calibration inputs and outputs must stay under ignored outputs/")
    if args.mode == "apply":
        rates = json.loads(args.calibration.read_text(encoding="utf-8"))["rates"]
        rows = _read_jsonl(args.predictions)
        if any(not row.get("valid") for row in rows):
            raise ValueError("invalid predictions cannot be calibrated or finalized")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                row["confidence"] = calibrated_confidence(row, rates)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"calibrated_predictions": len(rows)}, ensure_ascii=False))
        return
    expected = set()
    for view in "ABC":
        for row in _read_jsonl(args.data_dir / f"validation_{view}.jsonl"):
            expected.add((row["pack"], row["sample_id"], view))
    rows = _read_jsonl(args.validation_predictions)
    actual = [(row["pack"], row["sample_id"], row["view"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("calibration predictions must exactly cover the fixed validation views")
    gold = {}
    for pack in {row["pack"] for row in rows}:
        for item in core.read_json(args.train_root / pack / "gold" / "问答对_答案.json"):
            gold[(pack, item["sample_id"])] = item
    enriched = []
    for row in rows:
        if not row.get("valid"):
            continue
        metrics = score_row(row, gold[(row["pack"], row["sample_id"])])
        enriched.append({**row, "success": metrics["answer_char_f1"] >= .5
                         and metrics["chain_exact"] == 1})
    rates = fit_rates(enriched)
    nonrefusal = [row for row in enriched if row["answer"] != "无法确定"]
    squared_errors = []
    for fold in range(5):
        def fold_of(row):
            key = row["pack"].encode("utf-8")
            return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 5
        train = [row for row in nonrefusal if fold_of(row) != fold]
        test = [row for row in nonrefusal if fold_of(row) == fold]
        fold_rates = fit_rates(train)
        for row in test:
            probability = calibrated_confidence(row, fold_rates)
            squared_errors.append((probability - float(row["success"])) ** 2)
    report = {"source": "organizer TRAIN validation split only",
              "success_proxy": "answer_char_f1>=0.5 and a complete gold chain match",
              "calibrated_nonrefusal_predictions": len(nonrefusal),
              "package_grouped_5fold_brier_proxy": (
                  round(sum(squared_errors) / len(squared_errors), 5)
                  if squared_errors else None),
              "rates": rates,
              "note": "Beta(2,2) smoothing; length bucket used only with >=20 validation examples, otherwise view-level rate. No gold confidence levels became numeric training labels."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
