"""Run the refined four-way refusal prompt on frozen V4 diagnostic sets."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_deepseek_v4_staged_20261009 as runner  # noqa: E402


runner.OUT = ROOT / "outputs/deepseek_refusal_four_way_v6_20261010"
runner.ARM_FILES = {
    "v6": ROOT / "experiments/prompts/task4_refusal_four_way_v6_20261010.txt"
}
runner.SETS = {
    "old90": (ROOT / "experiments/reasoning_type_90_20261009.json", ("v6",)),
    # The original fresh30 manifest predates V6 and records only old-arm hashes.
    # The inherited runner records the new prompt hash in its own resume metadata.
    "fresh30_v6": (ROOT / "experiments/reasoning_type_fresh_30_20261009.json", ("v6",)),
}


if __name__ == "__main__":
    runner.main()
