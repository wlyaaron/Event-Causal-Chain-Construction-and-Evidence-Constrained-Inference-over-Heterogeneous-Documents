"""Offline Qwen3.5 thinking-mode pilot and blind run with bounded microbatches.

Pilot reads only organizer TRAIN validation inputs and gold for local scoring.
Blind mode reads only organizer test inputs. All outputs stay in ignored outputs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import task4_core as core
import task4_workflow as workflow
from experiments.run_full_causal_path_research import input_for as blind_input


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_after_thinking(raw: str) -> str:
    """Do not treat unfinished reasoning as an answer."""
    if "</think>" not in raw:
        raise ValueError("thinking did not close with </think>")
    final = raw.rsplit("</think>", 1)[1]
    for marker in ("<|im_end|>", "<|endoftext|>"):
        final = final.replace(marker, "")
    if not final.strip():
        raise ValueError("empty final answer after </think>")
    return final.strip()


def pilot_items(data_dir: Path, train_root: Path, split_file: Path,
                per_view: int) -> list[tuple[core.Sample, str]]:
    split = core.read_json(split_file)["splits"]
    validation_packs = set(split["validation"])
    samples = {(sample.pack.name, sample.sample_id): sample
               for sample in core.discover_samples(train_root, limit=0)}
    chosen = []
    for view in "ABC":
        rows = [json.loads(line) for line in
                (data_dir / f"validation_{view}.jsonl").open(encoding="utf-8")]
        selected = []
        # Include varied question types and an explanatory unanswerable case.
        for wanted in ("unanswerable_with_evidence", "counterfactual",
                       "retrospective", "prospective"):
            candidates = [row for row in rows if
                          (wanted in row["flags"] if wanted == "unanswerable_with_evidence"
                           else row["question_type"] == wanted)
                          and row["sample_id"] not in {r["sample_id"] for r in selected}]
            chosen_row = next((row for row in candidates if
                               row["pack"].rsplit("_", 1)[0] not in
                               {r["pack"].rsplit("_", 1)[0] for r in selected}), None)
            if chosen_row is None:
                chosen_row = next((row for row in candidates if row["pack"] not in
                                   {r["pack"] for r in selected}), None)
            if chosen_row is None and candidates:
                chosen_row = candidates[0]
            if chosen_row is not None:
                selected.append(chosen_row)
            if len(selected) >= per_view:
                break
        for row in rows:
            if len(selected) >= per_view:
                break
            if (row["sample_id"] not in {r["sample_id"] for r in selected}
                    and row["pack"] not in {r["pack"] for r in selected}):
                selected.append(row)
        for row in rows:
            if len(selected) >= per_view:
                break
            if row["sample_id"] not in {r["sample_id"] for r in selected}:
                selected.append(row)
        if len(selected) != per_view:
            raise ValueError(f"only {len(selected)} validation rows for view {view}")
        for row in selected:
            if row["pack"] not in validation_packs:
                raise ValueError("pilot row is outside fixed validation packs")
            chosen.append((samples[(row["pack"], row["sample_id"])], view))
    return chosen


def build_item(sample: core.Sample, view: str) -> tuple[str, set[str]]:
    if view == "blind":
        return blind_input(sample)
    if view == "A":
        content, allowed, _ = workflow.causal_path_input(sample, "A")
    else:
        content, allowed, _ = workflow.training_view_input(sample, view)
    return content, allowed


def make_batch(ids: list[list[int]], pad_token_id: int, device):
    import torch
    longest = max(map(len, ids))
    values = [[pad_token_id] * (longest - len(row)) + row for row in ids]
    masks = [[0] * (longest - len(row)) + [1] * len(row) for row in ids]
    return (torch.tensor(values, dtype=torch.long, device=device),
            torch.tensor(masks, dtype=torch.long, device=device))


def token_ids(encoded) -> list[int]:
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded


def generate(model, tokenizer, ids: list[list[int]], max_new_tokens: int,
             sampled: bool) -> tuple[list[str], list[int], float, list[list[int]]]:
    import torch
    tensor, mask = make_batch(ids, tokenizer.pad_token_id, model.device)
    settings = {"temperature": 1.0, "top_p": 0.95, "top_k": 20} if sampled else {}
    start = time.monotonic()
    with torch.inference_mode():
        result = model.generate(input_ids=tensor, attention_mask=mask,
                                max_new_tokens=max_new_tokens, do_sample=sampled,
                                pad_token_id=tokenizer.pad_token_id, **settings)
    elapsed = time.monotonic() - start
    texts, counts, tokens = [], [], []
    for row in result:
        output = row[tensor.shape[1]:].tolist()
        if tokenizer.eos_token_id in output:
            output = output[:output.index(tokenizer.eos_token_id) + 1]
        tokens.append(output)
        counts.append(len(output))
        texts.append(tokenizer.decode(output, skip_special_tokens=False))
    return texts, counts, elapsed, tokens


def generate_with_budget(model, tokenizer, ids: list[list[int]],
                         max_new_tokens: int, thinking_budget: int,
                         sampled: bool) -> tuple[list[str], list[int], float, list[bool], list[bool]]:
    if thinking_budget == 0:
        texts, counts, elapsed, _ = generate(model, tokenizer, ids,
                                             max_new_tokens, sampled)
        return texts, counts, elapsed, [False] * len(ids), [n >= max_new_tokens for n in counts]
    # This is an engineering adaptation of Qwen3's official two-pass thinking
    # budget recipe; Qwen3.5 support is verified on validation samples below.
    stop = tokenizer.encode(
        "\n\n时间有限，请基于已经核查的材料直接给出证据约束的结论。\n</think>\n\n",
        add_special_tokens=False)
    final_budget = max_new_tokens - thinking_budget - len(stop)
    if final_budget < 256:
        raise ValueError("output cap leaves fewer than 256 tokens for final JSON")
    first, counts, elapsed, first_tokens = generate(
        model, tokenizer, ids, thinking_budget, sampled)
    result = first.copy()
    forced = [False] * len(ids)
    cap = [False] * len(ids)
    continuations, positions = [], []
    for index, raw in enumerate(first):
        if "<|im_end|>" in raw or "<|endoftext|>" in raw:
            continue
        suffix = [] if "</think>" in raw else stop
        forced[index] = bool(suffix)
        continuations.append(ids[index] + first_tokens[index] + suffix)
        positions.append(index)
    if continuations:
        second, second_counts, second_seconds, _ = generate(
            model, tokenizer, continuations, final_budget, sampled)
        elapsed += second_seconds
        for position, raw, count in zip(positions, second, second_counts):
            result[position] += (tokenizer.decode(stop, skip_special_tokens=False)
                                 if forced[position] else "") + raw
            counts[position] += count
            cap[position] = count >= final_budget
    return result, counts, elapsed, forced, cap


def pilot_scores(items: list[tuple[core.Sample, str]], records: list[dict]) -> dict:
    from experiments.evaluate_sft import score_row
    rows = []
    for (sample, view), record in zip(items, records):
        gold = core.read_json(sample.pack / "gold" / "问答对_答案.json")
        reference = next(item for item in gold if item["sample_id"] == sample.sample_id)
        rows.append(score_row(record, reference))
    return {"samples": len(rows),
            "mean_answer_char_f1": round(sum(r["answer_char_f1"] for r in rows) / len(rows), 4),
            "mean_chain_event_f1": round(sum(r["chain_event_f1"] for r in rows) / len(rows), 4),
            "mean_chain_exact": round(sum(r["chain_exact"] for r in rows) / len(rows), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "blind"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--pilot-count-per-view", type=int, default=4)
    parser.add_argument("--data-dir", type=Path,
                        default=core.REPO_ROOT / "outputs" / "finetune_2026-10-06_v3")
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--split-file", type=Path,
                        default=core.REPO_ROOT / "experiments" / "finetune_split_20261005.json")
    parser.add_argument("--test-root", type=Path, default=core.DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--thinking-budget-hint", type=int, default=0,
                        help="ask the model to end its reasoning within this many tokens; not a hard cap")
    parser.add_argument("--thinking-budget-tokens", type=int, default=0,
                        help="hard two-pass reasoning budget; 0 lets the model think until output cap")
    parser.add_argument("--greedy", action="store_true",
                        help="diagnostic only; official Qwen thinking guidance uses sampling")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = (core.REPO_ROOT / "outputs").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or not args.model_dir.resolve().is_relative_to(root):
        raise ValueError("model and output must stay in ignored outputs/")
    if not 1 <= args.batch_size <= 4 or args.max_new_tokens < 256:
        raise ValueError("batch size must be 1..4 and output cap at least 256")
    if args.thinking_budget_hint < 0:
        raise ValueError("thinking budget hint cannot be negative")
    if args.thinking_budget_tokens < 0 or (args.thinking_budget_tokens and
                                            args.thinking_budget_tokens + 256 >= args.max_new_tokens):
        raise ValueError("thinking budget leaves too little space for final answer")
    if args.mode == "pilot":
        items = pilot_items(args.data_dir, args.train_root, args.split_file,
                            args.pilot_count_per_view)
    else:
        items = [(sample, "blind") for sample in
                 core.discover_samples(args.test_root, limit=0)]
        if len(items) != 760 or any(sample.track not in "ABC" for sample, _ in items):
            raise ValueError("blind run requires exactly 760 organizer test questions")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prompt = core.SYSTEM_PROMPT
    if args.thinking_budget_hint:
        system_prompt += (f"\n请优先核查与问题直接相关的证据，尽量在"
                          f"{args.thinking_budget_hint} 个 token 内结束思考，"
                          "随后给出完整的最终 JSON。不要省略最终答案。")
    prepared = []
    for sample, view in items:
        content, allowed = build_item(sample, view)
        ids = token_ids(tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True))
        if len(ids) + args.max_new_tokens > args.max_context_tokens:
            raise ValueError(f"{sample.sample_id}/{view}: {len(ids)} input tokens plus "
                             f"{args.max_new_tokens} output budget exceed context; no truncation")
        prepared.append((ids, allowed))
    signature = {"mode": args.mode, "sample_views_sha256": hashlib.sha256(
        "\n".join(f"{sample.sample_id}/{view}" for sample, view in items).encode()).hexdigest(),
                 "model_config_sha256": sha256(args.model_dir / "config.json"),
                 "tokenizer_sha256": sha256(args.model_dir / "tokenizer.json"),
                 "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
                 "max_context_tokens": args.max_context_tokens,
                 "seed": args.seed, "sampled": not args.greedy,
                 "thinking_budget_hint": args.thinking_budget_hint,
                 "thinking_budget_tokens": args.thinking_budget_tokens}
    print(json.dumps({"audit": signature, "items": len(items),
                      "max_prompt_tokens": max(len(row[0]) for row in prepared)},
                     ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if output.exists():
        raise FileExistsError(output)
    progress = output.with_name(output.name + ".progress.json")
    if progress.exists():
        state = core.read_json(progress)
        if state["signature"] != signature:
            raise ValueError("progress signature mismatch")
    else:
        state = {"signature": signature, "records": [], "details": []}
        core.atomic_json(progress, state)
    if len(state["records"]) != len(state["details"]):
        raise ValueError("progress records/details mismatch")
    for index, record in enumerate(state["records"]):
        sample, _ = items[index]
        if core.parse_prediction(json.dumps(record, ensure_ascii=False),
                                 sample, prepared[index][1]) != record:
            raise ValueError("invalid progress record")

    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa").eval()
    print("Qwen3.5 thinking model loaded; offline local weights only", flush=True)
    while len(state["records"]) < len(items):
        start = len(state["records"])
        end = min(start + args.batch_size, len(items))
        texts, counts, elapsed, forced, cap_hits = generate_with_budget(
            model, tokenizer, [prepared[i][0] for i in range(start, end)],
            args.max_new_tokens, args.thinking_budget_tokens, not args.greedy)
        for offset, index in enumerate(range(start, end)):
            sample, view = items[index]
            errors = []
            raw = texts[offset]
            try:
                record = core.parse_prediction(final_after_thinking(raw),
                                               sample, prepared[index][1])
            except ValueError as exc:
                errors.append(str(exc))
                # Preserve the original answer as a diagnostic; repair output
                # format with a single direct-response pass on the same input.
                repaired_content, _ = build_item(sample, view)
                repaired_content += ("\n\n请只给出符合指定字段的完整 JSON；"
                                     "证据 ID 必须出自原材料。")
                repaired_ids = token_ids(tokenizer.apply_chat_template(
                    [{"role": "system", "content": core.SYSTEM_PROMPT},
                     {"role": "user", "content": repaired_content}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False))
                if len(repaired_ids) + args.max_new_tokens > args.max_context_tokens:
                    raise ValueError("format repair would truncate input")
                retry, retry_counts, retry_seconds, _ = generate(
                    model, tokenizer, [repaired_ids], args.max_new_tokens, False)
                counts[offset] += retry_counts[0]
                elapsed += retry_seconds
                try:
                    cleaned = retry[0].replace("<|im_end|>", "").replace("<|endoftext|>", "")
                    record = core.parse_prediction(cleaned, sample, prepared[index][1])
                except ValueError as retry_exc:
                    errors.append(str(retry_exc))
                    record = core.parse_prediction(
                        '{"answer":"无法确定","evidence_chain":[],"confidence":null}',
                        sample, prepared[index][1])
            state["records"].append(record)
            state["details"].append({
                "sample_id": sample.sample_id, "view": view,
                "prompt_tokens": len(prepared[index][0]),
                "generated_tokens": counts[offset],
                "batch_seconds": 0.0,
                "thinking_closed": "</think>" in raw,
                "forced_thinking_end": forced[offset],
                "hit_output_cap": cap_hits[offset],
                "format_errors": errors, "format_fallback": len(errors) == 2})
        state["details"][start]["batch_seconds"] = round(elapsed, 3)
        core.atomic_json(progress, state)
        print(f"[{end}/{len(items)}] batch={args.batch_size} "
              f"seconds={elapsed:.2f} errors={sum(bool(d['format_errors']) for d in state['details'][start:end])}",
              flush=True)
    core.atomic_json(output, state["records"])
    if args.mode == "blind":
        core.validate_file(output, [sample for sample, _ in items], 80000)
    else:
        report = pilot_scores(items, state["records"])
        report.update({"format_fallbacks": sum(d["format_fallback"] for d in state["details"]),
                       "unfinished_thinking": sum(not d["thinking_closed"] for d in state["details"]),
                       "output_cap_hits": sum(d["hit_output_cap"] for d in state["details"]),
                       "total_generated_tokens": sum(d["generated_tokens"] for d in state["details"]),
                       "total_batch_seconds": round(sum(d["batch_seconds"] for d in state["details"]), 2)})
        core.atomic_json(output.with_name(output.stem + ".report.json"), report)
    print(json.dumps({"status": "complete", "samples": len(items),
                      "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
