"""Score the chain-only ablation with the frozen 50 and intact gold."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import score_a_candidate_answer_aligned_v2_20261010 as scorer


scorer.OUT = ROOT / "outputs/a_candidate_chain_only_20261010"
scorer.SCORE = scorer.OUT / "score"


if __name__ == "__main__":
    scorer.main()
