"""Run the frozen blind-test workflow with local Qwen3.5 weights, without an API.

This consumes only organizer test inputs. Test gold and hosted-model outputs are
never read. A uses the frozen question-conditioned path input; B/C use the
original input. Results remain under ignored outputs/ and are not submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import task4_core as core
from experiments.run_full_causal_path_research import input_for


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def messages(content: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": core.SYSTEM_PROMPT},
            {"role": "user", "content": content}]


def encode(tokenizer, content: str):
    return tokenizer.apply_chat_template(
        messages(content), tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt")


def token_ids(encoded):
    return encoded["input_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=core.DEFAULT_DATASET)
    parser.add_argument("--track", choices=["all", "A", "B", "C"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    ignored = (core.REPO_ROOT / "outputs").resolve()
    if not output.is_relative_to(ignored):
        raise ValueError("output must stay under ignored outputs/")
    if args.max_new_tokens < 256 or args.max_context_tokens <= args.max_new_tokens:
        raise ValueError("invalid context or output-token limit")
    samples = core.discover_samples(args.dataset, args.track, args.limit)
    if any(sample.track not in "ABC" for sample in samples):
        raise ValueError("blind-test A/B/C samples only")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompts: list[tuple[str, set[str], int]] = []
    counts = Counter()
    for sample in samples:
        content, allowed = input_for(sample)
        length = token_ids(encode(tokenizer, content)).shape[-1]
        if length + args.max_new_tokens > args.max_context_tokens:
            raise ValueError(f"{sample.sample_id}: {length} prompt tokens plus output cap "
                             f"exceeds {args.max_context_tokens}; no input was truncated")
        prompts.append((content, allowed, length))
        counts[sample.track] += 1
    audit = {"samples": len(samples), "by_track": dict(counts),
             "max_prompt_tokens": max(length for _, _, length in prompts),
             "mean_prompt_tokens": round(sum(length for _, _, length in prompts) / len(prompts), 1),
             "max_context_tokens": args.max_context_tokens,
             "max_new_tokens": args.max_new_tokens}
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this 9B local run")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    metadata = {
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": file_sha256(args.model_dir / "config.json"),
        "tokenizer_sha256": file_sha256(args.model_dir / "tokenizer.json"),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sample.sample_id for sample in samples).encode()).hexdigest(),
        "track": args.track, "limit": args.limit,
        "prompt_route": "A:causal_path_input; B/C:core.build_input",
        "chat_template_enable_thinking": False, "do_sample": False,
        "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": args.max_new_tokens,
    }
    store = core.PredictionStore(output, samples, metadata)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa").eval()
    print("Local Qwen3.5 loaded on CUDA; no API endpoint is used.", flush=True)

    details_path = output.with_name(output.stem + ".details.progress.json")
    details = core.read_json(details_path) if details_path.exists() else []
    if len(details) > len(store.completed):
        details = details[:len(store.completed)]
    for index in range(len(store.completed), len(samples)):
        sample = samples[index]
        content, allowed, prompt_length = prompts[index]
        start = time.monotonic()
        errors = []
        for correction in range(2):
            encoded = encode(tokenizer, content)
            ids = token_ids(encoded)
            if ids.shape[-1] + args.max_new_tokens > args.max_context_tokens:
                raise ValueError(f"{sample.sample_id}: correction would exceed context; no truncation")
            ids = ids.to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            new_ids = generated[0, ids.shape[-1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            try:
                record = core.parse_prediction(raw, sample, allowed)
                break
            except ValueError as exc:
                errors.append(str(exc))
                if correction:
                    record = core.parse_prediction(
                        '{"answer":"无法确定","evidence_chain":[],"confidence":null}',
                        sample, allowed)
                    break
                content += ("\n\n上一版输出未通过格式校验：" + str(exc)
                            + "。请仅输出修正后的完整 JSON，证据链只用真实事件 ID。")
        store.append(record)
        details.append({"sample_id": sample.sample_id, "track": sample.track,
                        "prompt_tokens": prompt_length, "generated_tokens": len(new_ids),
                        "seconds": round(time.monotonic() - start, 2),
                        "format_errors": errors, "format_fallback": len(errors) == 2})
        core.atomic_json(details_path, details)
        print(f"[{index + 1}/{len(samples)}] {sample.sample_id} "
              f"{details[-1]['seconds']}s errors={len(errors)}", flush=True)
    store.finish()
    print(json.dumps({"status": "complete", "samples": len(samples),
                      "output": str(output), "details": str(details_path)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
