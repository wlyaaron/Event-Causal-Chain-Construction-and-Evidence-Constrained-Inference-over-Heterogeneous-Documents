"""Run the four-way refusal prompt on the existing old-90 development set.

Uses the same frozen inputs, API wrapper, model and 8192-token cap as V4.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_deepseek_v4_staged_20261009 as runner


runner.OUT = ROOT / "outputs/deepseek_refusal_four_way_v5_20261010"
runner.ARM_FILES = {
    "v5": ROOT / "experiments/prompts/task4_refusal_four_way_v5_20261010.txt"
}
runner.SETS = {
    "old90": (ROOT / "experiments/reasoning_type_90_20261009.json", ("v5",))
}


if __name__ == "__main__":
    runner.main()
