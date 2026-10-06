"""Register organizer TRAIN SFT rows for pinned LLaMA-Factory without truncation.

Lengths are computed with the actual LLaMA-Factory template used for training.
Rows exceeding cutoff_len are recorded and excluded before its preprocessor,
which otherwise truncates source/target tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import task4_core as core


TEMPLATE_NAME = "qwen3_5_nothink"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-len", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--llamafactory-commit", required=True)
    parser.add_argument("--smoke-per-view", type=int, default=0)
    parser.add_argument("--longest-per-view", type=int, default=0)
    args = parser.parse_args()

    ignored = (core.REPO_ROOT / "outputs").resolve()
    for path in (args.source, args.tokenizer, args.output_dir):
        if not path.resolve().is_relative_to(ignored):
            raise ValueError(f"all paths must remain in ignored outputs/: {path}")
    if not args.source.name.startswith("fit_"):
        raise ValueError("only fit training rows may be registered")
    if args.cutoff_len < 256 or args.smoke_per_view < 0 or args.longest_per_view < 0:
        raise ValueError("invalid length or smoke count")
    if args.smoke_per_view and args.longest_per_view:
        raise ValueError("select shortest or longest samples, not both")

    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    template = TEMPLATES[TEMPLATE_NAME]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset_name}.jsonl"
    if output_file.exists():
        raise FileExistsError(output_file)
    temp_file = output_dir / f".{args.dataset_name}.jsonl.tmp"
    excluded = []
    eligible_by_view = Counter()
    written_by_view = Counter()
    shortest: dict[str, list[tuple[int, str, dict]]] = {view: [] for view in "ABC"}
    seen_smoke = set()
    maximum_written = 0
    source_rows = 0
    with args.source.open(encoding="utf-8") as source, temp_file.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            source_rows += 1
            if row.get("view") not in "ABC" or not row.get("pack") or not row.get("sample_id"):
                raise ValueError("missing view, pack, or sample_id")
            messages = row["messages"]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError("expected system/user/assistant")
            prompt_ids, answer_ids = template.encode_oneturn(
                tokenizer, messages[1:], system=messages[0]["content"]
            )
            if not answer_ids:
                raise ValueError("empty supervised answer")
            length = len(prompt_ids) + len(answer_ids)
            if length > args.cutoff_len:
                excluded.append({"pack": row["pack"], "sample_id": row["sample_id"],
                                 "view": row["view"], "tokens": length})
                continue
            eligible_by_view[row["view"]] += 1
            if args.smoke_per_view or args.longest_per_view:
                key = (row["pack"], row["sample_id"], row["view"])
                if key in seen_smoke:
                    continue
                seen_smoke.add(key)
                candidates = shortest[row["view"]]
                candidates.append((length, f"{row['pack']}:{row['sample_id']}", row))
                candidates.sort(key=lambda item: (-item[0], item[1]) if args.longest_per_view else (item[0], item[1]))
                del candidates[(args.longest_per_view or args.smoke_per_view):]
            else:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_by_view[row["view"]] += 1
                maximum_written = max(maximum_written, length)
        if args.smoke_per_view or args.longest_per_view:
            for view in "ABC":
                for length, _, row in shortest[view]:
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written_by_view[view] += 1
                    maximum_written = max(maximum_written, length)
    temp_file.replace(output_file)

    registration = {
        args.dataset_name: {
            "file_name": output_file.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant",
                     "system_tag": "system"},
        }
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(registration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": str(args.source), "source_sha256": sha256(args.source),
        "tokenizer": str(args.tokenizer),
        "tokenizer_json_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "llamafactory_commit": args.llamafactory_commit,
        "template": TEMPLATE_NAME, "cutoff_len": args.cutoff_len,
        "source_rows": source_rows, "eligible_by_view": dict(eligible_by_view),
        "written_by_view": dict(written_by_view),
        "max_written_tokens": maximum_written,
        "excluded_overlength": excluded,
        "dataset_sha256": sha256(output_file),
        "note": "Excluded rows were not truncated; full original source remains available.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"written": sum(written_by_view.values()),
                      "by_view": dict(written_by_view), "overlength": len(excluded),
                      "max_tokens": maximum_written, "dataset": str(output_file)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
