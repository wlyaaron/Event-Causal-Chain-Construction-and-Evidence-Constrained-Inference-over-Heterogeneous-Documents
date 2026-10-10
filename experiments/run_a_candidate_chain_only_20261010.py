"""Run the frozen 50 with the v2 answer paragraph removed.

This reuses the frozen runner and its input/format/budget checks exactly.
Only the prompt file and output directory change; gold stays out of requests.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_a_candidate_answer_aligned_v2_20261010 as runner


runner.PREFIX = ROOT / "experiments/prompts/task4_a_candidate_chain_only_20261010.txt"
runner.OUT = ROOT / "outputs/a_candidate_chain_only_20261010"


if __name__ == "__main__":
    runner.main()
