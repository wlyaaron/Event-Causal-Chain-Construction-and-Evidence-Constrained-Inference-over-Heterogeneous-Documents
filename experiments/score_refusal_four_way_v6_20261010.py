"""Score V6 with intact gold and frozen denominators."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import score_deepseek_v4_staged_20261009 as scorer  # noqa: E402


scorer.OUT = ROOT / "outputs/deepseek_refusal_four_way_v6_20261010"
scorer.SCORE = scorer.OUT / "score"
scorer.MANIFESTS = {
    "old90": ROOT / "experiments/reasoning_type_90_20261009.json",
    "fresh30_v6": ROOT / "experiments/reasoning_type_fresh_30_20261009.json",
}
scorer.ARMS = {"old90": ("v6",), "fresh30_v6": ("v6",)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-fresh30", action="store_true")
    args = parser.parse_args()
    if not args.include_fresh30:
        scorer.MANIFESTS = {"old90": scorer.MANIFESTS["old90"]}
        scorer.ARMS = {"old90": scorer.ARMS["old90"]}
    scorer.main()
