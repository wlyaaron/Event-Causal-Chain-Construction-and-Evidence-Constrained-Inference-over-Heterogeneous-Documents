"""Freeze previous guided-arm invalid outputs for a gold-blind repair diagnostic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task4_core as core  # noqa: E402
from experiments.freeze_a_candidate_v4_pair_20261010 import (  # noqa: E402
    OUT as ORIGINAL_MANIFEST, PROMPT as V4_PROMPT, sha,
)

ORIGINAL_ATTEMPTS = (ROOT / "outputs" / "a_candidate_v4_pair_holdout50_20261010" /
                     "attempts.jsonl")
REPAIR_PREFIX = (ROOT / "experiments" / "prompts" /
                 "task4_a_candidate_repair_v1_20261010.txt")
OUT = ROOT / "experiments" / "a_candidate_v4_repair6_20261010.json"


def main() -> None:
    old = core.read_json(ORIGINAL_MANIFEST)
    attempts = [json.loads(line) for line in ORIGINAL_ATTEMPTS.read_text(
        encoding="utf-8").splitlines()]
    invalid = {item["sample_id"]: item for item in attempts
               if item["arm"] == "guided" and item["parsed"] is None}
    rows = []
    for row in old["rows"]:
        sid = row["sample_id"]
        if sid in invalid:
            attempt = invalid[sid]["attempt"]
            reason = ("truncated" if attempt.get("finish_reason") == "length"
                      else "schema_invalid")
            rows.append({**row, "previous_failure_type": reason})
    if len(rows) != 6 or len({r["sample_id"] for r in rows}) != 6:
        raise ValueError("Expected exactly six prior invalid guided responses")
    result = {"selection": "all six previously invalid guided outputs, no gold selection",
              "source_manifest_sha256": sha(ORIGINAL_MANIFEST.read_bytes()),
              "source_attempts_sha256": sha(ORIGINAL_ATTEMPTS.read_bytes()),
              "v4_prompt_sha256": sha(V4_PROMPT.read_bytes()),
              "repair_prefix_sha256": sha(REPAIR_PREFIX.read_bytes()),
              "gold_used_for_selection": False, "rows": rows}
    if OUT.exists() and core.read_json(OUT) != result:
        raise ValueError("Frozen repair manifest differs; refusing overwrite")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"path": str(OUT), "n": len(rows),
                      "failure_types": {kind: sum(r["previous_failure_type"] == kind
                                                  for r in rows)
                                        for kind in ("schema_invalid", "truncated")}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
