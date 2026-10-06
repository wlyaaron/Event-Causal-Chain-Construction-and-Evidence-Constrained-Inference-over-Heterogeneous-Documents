"""Offline LoRA inference on fixed TRAIN validation or holdout packages.

Only source materials and questions enter the model. Output has no numeric
confidence until validation-set calibration has been measured separately.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import task4_core as core
import task4_workflow as workflow
from experiments.prepare_finetune_data import SFT_SYSTEM_PROMPT


def validate_answer_chain(raw: str, allowed: set[str]) -> dict:
    value = json.loads(raw.strip())
    if not isinstance(value, dict):
        raise ValueError("prediction is not a JSON object")
    if set(value) != {"answer", "evidence_chain"}:
        raise ValueError("prediction must contain exactly answer and evidence_chain")
    answer, chain = value.get("answer"), value.get("evidence_chain")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer is empty")
    if not isinstance(chain, list) or any(not isinstance(node, str) for node in chain):
        raise ValueError("evidence_chain must be an array of IDs")
    if len(chain) != len(set(chain)):
        raise ValueError("duplicate evidence IDs")
    if not set(chain) <= allowed:
        raise ValueError("unknown evidence IDs")
    if answer.strip() == "无法确定" and chain:
        raise ValueError("refusal has a nonempty evidence chain")
    if answer.strip() != "无法确定" and not chain:
        raise ValueError("non-refusal has an empty evidence chain")
    return {"answer": answer.strip(), "evidence_chain": chain}


def load_items(train_root: Path, data_dir: Path, split: str):
    keys = set()
    for view in "ABC":
        with (data_dir / f"{split}_{view}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                keys.add((row["pack"], row["sample_id"], view))
    samples = {(sample.pack.name, sample.sample_id): sample
               for sample in core.discover_samples(train_root, limit=0)}
    for pack, sample_id, view in sorted(keys):
        sample = samples[(pack, sample_id)]
        content, allowed, _ = workflow.training_view_input(sample, view)
        yield (pack, sample_id, view, content, allowed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "holdout"], required=True)
    parser.add_argument("--train-root", type=Path,
                        default=core.REPO_ROOT / "数据集" / "训练集")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ignored = (core.REPO_ROOT / "outputs").resolve()
    for path in (args.data_dir, args.model_dir, args.tokenizer_dir, args.output):
        if not path.resolve().is_relative_to(ignored):
            raise ValueError("model, data and output paths must stay under ignored outputs/")
    if args.adapter_dir and not args.adapter_dir.resolve().is_relative_to(ignored):
        raise ValueError("adapter must stay under ignored outputs/")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    items = list(load_items(args.train_root, args.data_dir, args.split))
    prepared, over = [], []
    for pack, sample_id, view, content, allowed in items:
        prompt = [{"role": "system", "content": SFT_SYSTEM_PROMPT},
                  {"role": "user", "content": content}]
        ids = tokenizer.apply_chat_template(prompt, tokenize=True,
                                            add_generation_prompt=True)
        if len(ids) + args.max_new_tokens > args.max_context_tokens:
            over.append({"pack": pack, "sample_id": sample_id, "view": view,
                         "input_tokens": len(ids)})
        prepared.append((pack, sample_id, view, content, allowed, len(ids)))
    if args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"items": len(prepared),
                                           "over_context": over,
                                           "quantization": args.quantization,
                                           "max_context_tokens": args.max_context_tokens,
                                           "max_new_tokens": args.max_new_tokens},
                                          ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(json.dumps({"items": len(prepared), "over_context": len(over)},
                         ensure_ascii=False))
        return
    if over:
        raise ValueError(f"{len(over)} prompts exceed context budget; inspect with --check-only")
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; inference not started")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = {"local_files_only": True, "torch_dtype": dtype,
                    "device_map": {"": torch.cuda.current_device()}}
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **model_kwargs)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir,
                                          local_files_only=True)
    model.eval()
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    metadata = partial.with_suffix(partial.suffix + ".meta.json")
    signature = {"split": args.split, "model_dir": str(args.model_dir.resolve()),
                 "adapter_dir": str(args.adapter_dir.resolve()) if args.adapter_dir else None,
                 "quantization": args.quantization,
                 "max_context_tokens": args.max_context_tokens,
                 "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                 "expected_records": len(prepared)}
    done = set()
    if partial.exists():
        if not metadata.exists() or core.read_json(metadata) != signature:
            raise ValueError("partial prediction metadata mismatch")
        with partial.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                done.add((row["pack"], row["sample_id"], row["view"]))
    else:
        if metadata.exists():
            raise ValueError("orphaned partial metadata")
        core.atomic_json(metadata, signature)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a", encoding="utf-8", newline="\n") as output:
        for pack, sample_id, view, content, allowed, checked_tokens in prepared:
            if (pack, sample_id, view) in done:
                continue
            start = time.perf_counter()
            prompt = [{"role": "system", "content": SFT_SYSTEM_PROMPT},
                      {"role": "user", "content": content}]
            input_ids = tokenizer.apply_chat_template(prompt, tokenize=True,
                                                       add_generation_prompt=True)
            if len(input_ids) != checked_tokens:
                raise ValueError("token count changed since context check")
            tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                generated = model.generate(input_ids=tensor, do_sample=False,
                                           max_new_tokens=args.max_new_tokens,
                                           pad_token_id=tokenizer.eos_token_id)
            new_ids = generated[0, len(input_ids):].tolist()
            raw = tokenizer.decode(new_ids, skip_special_tokens=True)
            try:
                parsed = validate_answer_chain(raw, allowed)
                valid, error = True, None
            except (ValueError, json.JSONDecodeError) as exc:
                parsed = {"answer": "", "evidence_chain": []}
                valid, error = False, str(exc)
            record = {"pack": pack, "sample_id": sample_id, "view": view,
                      **parsed, "valid": valid, "error": error,
                      "input_tokens": len(input_ids), "output_tokens": len(new_ids),
                      "latency_seconds": round(time.perf_counter() - start, 3)}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    os.replace(partial, args.output)
    os.replace(metadata, args.output.with_suffix(args.output.suffix + ".meta.json"))
    print(json.dumps({"predictions": len(prepared), "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
