"""Research-only paired workflow experiment on labeled training questions.

The model receives source documents and retrieval hints, never gold answers.
Outputs under outputs/ are excluded from Git and are not competition submissions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import task4_api as api
import task4_core as core
import task4_workflow as workflow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, required=True,
                        help="UTF-8 JSON array of training sample IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--api-url", default=api.DEFAULT_API_URL)
    parser.add_argument("--model", default=api.DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["guided", "guided_verify"],
                        default="guided")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hint-mode", choices=["direct", "nodes", "graph_paths",
                                                "learned"],
                        default="nodes")
    parser.add_argument("--ranker-checkpoint", type=Path,
                        help="Local checkpoint produced by train_event_retriever")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    ids = json.loads(args.ids_file.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("ids-file must contain unique sample IDs")
    available = {sample.sample_id: sample for sample in core.discover_samples(
        core.REPO_ROOT / "数据集" / "训练集", limit=0)}
    if missing := set(ids) - set(available):
        raise ValueError(f"IDs absent from training set: {sorted(missing)[:3]}")
    samples = [available[sample_id] for sample_id in ids]
    ranker = None
    ranker_features = None
    if args.hint_mode == "learned":
        if args.ranker_checkpoint is None:
            raise ValueError("learned hint mode requires --ranker-checkpoint")
        import joblib
        import numpy as np
        import run_baselines as baseline
        from experiments.train_event_retriever import features
        ranker = joblib.load(args.ranker_checkpoint)
        ranker_features = features
    endpoint, key, model = api.read_api_settings(args)
    audit_path = args.output.with_suffix(".audit.json")
    records = core.read_json(args.output) if args.output.exists() else []
    audit = core.read_json(audit_path) if audit_path.exists() else []
    checkpoint_path = (str(args.ranker_checkpoint.resolve())
                       if args.ranker_checkpoint else None)
    if len(records) != len(audit) or len(records) > len(samples):
        raise ValueError("Existing output and audit do not form a valid checkpoint")
    if any(row.get("sample_id") != sample.sample_id
           or note.get("sample_id") != sample.sample_id
           or note.get("mode") != args.mode
           or note.get("hint_mode", "nodes") != args.hint_mode
           or note.get("ranker_checkpoint") != checkpoint_path
           for row, note, sample in zip(records, audit, samples)):
        raise ValueError("Existing checkpoint IDs or mode do not match this experiment")
    if records:
        print(f"resuming after {len(records)} saved records", flush=True)

    def predict(content: str, sample: core.Sample,
                allowed_ids: set[str]) -> tuple[dict, bool]:
        for attempt in range(2):
            output_cap = args.max_tokens
            for expansion in range(3):
                try:
                    raw = api.call_model(endpoint, key, model, content,
                                         timeout=args.timeout, max_tokens=output_cap)
                    break
                except ValueError as exc:
                    if "max_tokens 截断" not in str(exc) or expansion == 2:
                        raise
                    output_cap = min(output_cap * 2, 32768)
            try:
                return core.parse_prediction(raw, sample, allowed_ids), False
            except ValueError as exc:
                if attempt:
                    fallback = '{"answer":"无法确定","evidence_chain":[],"confidence":null}'
                    print(f"{sample.sample_id}: two invalid formats; conservative fallback: {exc}",
                          file=sys.stderr, flush=True)
                    return core.parse_prediction(fallback, sample, allowed_ids), True
                content += ("\n\n上次输出不符合 JSON 提交约束：" + str(exc)
                            + "。请核对原文，只返回合法 JSON。")
        raise AssertionError("unreachable")

    for index, sample in enumerate(samples[len(records):], start=len(records) + 1):
        ranked_ids = None
        if ranker is not None:
            pack = baseline.read_pack(sample.pack)
            candidate_ids, matrix, _ = ranker_features(
                sample, pack, True, ranker["ablation"])
            scores = ranker["model"].predict_proba(matrix)[:, 1]
            ranked_ids = [candidate_ids[position] for position in np.argsort(-scores)]
        guided, hints = workflow.guided_input(sample, args.top_k,
                                               hint_mode=args.hint_mode,
                                               ranked_ids=ranked_ids)
        _, allowed_ids, _ = core.build_input(sample)
        started = time.monotonic()
        initial, initial_fallback = predict(guided, sample, allowed_ids)
        final = initial
        final_fallback = initial_fallback
        if args.mode == "guided_verify":
            verification = (
                guided + "\n\n待核验的第一版回答：\n"
                + json.dumps(initial, ensure_ascii=False)
                + "\n\n请逐项核对答案中的事实、证据链每一步的方向和拒答条件。"
                  "仅在原文证据支持时保留；发现缺漏或材料外断言则修订。"
                  "只输出最终的 answer、evidence_chain、confidence JSON 对象。"
            )
            final, final_fallback = predict(verification, sample, allowed_ids)
        records.append(final)
        audit.append({"sample_id": sample.sample_id,
                      "mode": args.mode, "model": model,
                      "hint_mode": args.hint_mode,
                      "ranker_checkpoint": checkpoint_path,
                      "input_chars": len(guided),
                      "hint_ids": [hint.event_id for hint in hints],
                      "initial": initial,
                      "initial_format_fallback": initial_fallback,
                      "final_format_fallback": final_fallback,
                      "changed_by_verification": final != initial,
                      "elapsed_seconds": round(time.monotonic() - started, 2)})
        core.atomic_json(args.output, records)
        core.atomic_json(audit_path, audit)
        print(f"[{index}/{len(samples)}] {sample.sample_id} saved", flush=True)
    print(f"Research predictions: {args.output} ({len(records)} training questions)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
