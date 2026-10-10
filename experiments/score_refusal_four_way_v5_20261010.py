"""Score V5 on the old-90 development set with original complete gold."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import score_deepseek_v4_staged_20261009 as scorer


scorer.OUT = ROOT / "outputs/deepseek_refusal_four_way_v5_20261010"
scorer.SCORE = scorer.OUT / "score"
scorer.MANIFESTS = {
    "old90": ROOT / "experiments/reasoning_type_90_20261009.json"
}
scorer.ARMS = {"old90": ("v5",)}


if __name__ == "__main__":
    scorer.main()
