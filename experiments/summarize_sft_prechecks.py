"""Collect all seven offline SFT input checks without loading a GPU model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import task4_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path,
                        default=core.REPO_ROOT / "experiments" / "configs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((core.REPO_ROOT / "outputs").resolve()):
        raise ValueError("summary must stay under ignored outputs/")
    reports = {}
    for config_path in sorted(args.config_dir.glob("task4_qwen3_4b_*.json")):
        config = core.read_json(config_path)
        manifest_path = Path(config["output_dir"]) / "input_manifest.json"
        manifest = core.read_json(manifest_path)
        if (manifest["model_revision"] != config["model_revision"] or
                manifest["configuration"] != config):
            raise ValueError(f"stale input manifest: {config_path}")
        excluded = manifest["excluded_rows"]
        included = manifest["trained_rows"]
        total = included + len(excluded)
        reports[config_path.stem] = {
            "source_jsonl": config["train_jsonl"],
            "source_sha256": manifest["train_jsonl_sha256"],
            "max_seq_tokens": config["max_seq_tokens"],
            "input_rows": total,
            "eligible_rows": included,
            "excluded_overlength_rows": len(excluded),
            "excluded_by_view": dict(sorted(Counter(row["view"] for row in excluded).items())),
            "excluded_percent": round(100 * len(excluded) / total, 2),
            "eligible_input_tokens_once": manifest["eligible_input_tokens_once"],
            "eligible_supervised_tokens_once": manifest["eligible_supervised_tokens_once"],
            "length_cache_hits": manifest.get("length_cache_hits"),
        }
    if len(reports) != 7:
        raise ValueError(f"expected seven SFT configurations, found {len(reports)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    core.atomic_json(args.output, reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
