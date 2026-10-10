"""One-call paired DeepSeek V4 versus V4 plus concise format instructions."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
from experiments import run_deepseek_v4_staged_20261009 as runner  # noqa: E402
from experiments.run_codex_initial_deepseek_protocol import sha  # noqa: E402


runner.OUT = ROOT / "outputs/deepseek_refusal_format24_20261010"
runner.ARM_FILES = {
    "v4": ROOT / "experiments/prompts/task4_merged_rules_v4_draft_20261009.txt",
    "fmt": ROOT / "experiments/prompts/task4_refusal_format_v7_20261010.txt",
}
runner.SETS = {
    "format24": (ROOT / "experiments/refusal_format24_20261010.json",
                 ("v4", "fmt")),
}


if __name__ == "__main__":
    frozen = core.read_json(runner.SETS["format24"][0])
    if any(frozen["prompt_sha256"][arm] != sha(path.read_bytes())
           for arm, path in runner.ARM_FILES.items()):
        raise ValueError("Prompt differs from frozen refusal-format manifest")
    runner.main()
