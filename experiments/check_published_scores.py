"""Reconcile published A/B/C scores with the published 60/30/10 rule.

This does not score hidden test answers. It only audits numbers returned by the
platform or printed in an organizer baseline table.
"""

from __future__ import annotations

import argparse
import json


WEIGHTS = {"A": 0.6, "B": 0.3, "C": 0.1}


def weighted(scores: dict[str, float]) -> float:
    return sum(WEIGHTS[track] * scores[track] for track in WEIGHTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for track in WEIGHTS:
        parser.add_argument(f"--{track.lower()}", type=float, required=True)
    parser.add_argument("--reported-total", type=float)
    args = parser.parse_args()
    scores = {"A": args.a, "B": args.b, "C": args.c}
    calculated = weighted(scores)
    result = {"track_scores": scores, "weights": WEIGHTS,
              "calculated_total": round(calculated, 4)}
    if args.reported_total is not None:
        result["reported_total"] = args.reported_total
        result["reported_minus_calculated"] = round(args.reported_total - calculated, 4)
        result["consistent_to_two_decimals"] = round(calculated, 2) == args.reported_total
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
