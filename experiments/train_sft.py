"""Single-GPU LoRA SFT with assistant-only loss and explicit length exclusions.

The check-only path needs no GPU or model weights. Training fails if any input
would be truncated unless the config explicitly allows and records its omission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import task4_core as core


def encode_supervised(row: dict, tokenizer, max_tokens: int) -> dict:
    messages = row["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected a system/user/assistant training record")
    prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False,
                                                add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template prompt is not a text prefix")
    encoded = tokenizer(full_text, add_special_tokens=False,
                        return_offsets_mapping=True)
    full = encoded["input_ids"]
    boundary = len(prompt_text)
    if any(start < boundary < end for start, end in encoded["offset_mapping"]):
        raise ValueError("a token crosses the assistant target boundary")
    prompt_length = sum(end <= boundary for _, end in encoded["offset_mapping"])
    if len(full) > max_tokens:
        raise ValueError(f"sequence exceeds {max_tokens} tokens: {len(full)}")
    if len(full) == prompt_length:
        raise ValueError("assistant target is empty")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * prompt_length + full[prompt_length:]}


def prepare_records(source: Path, tokenizer, max_tokens: int,
                    allow_skip: bool, cache: sqlite3.Connection,
                    tokenizer_fingerprint: str) -> tuple[list[int], list[dict], int, int, int]:
    offsets, excluded = [], []
    input_tokens, supervised_tokens = 0, 0
    cache_hits = 0
    with source.open(encoding="utf-8") as input_file:
        while True:
            offset = input_file.tell()
            line = input_file.readline()
            if not line:
                break
            row = json.loads(line)
            message_bytes = json.dumps(row["messages"], ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8")
            key = hashlib.sha256(tokenizer_fingerprint.encode("ascii") + b"\0" +
                                 message_bytes).hexdigest()
            cached = cache.execute("SELECT full_tokens, target_tokens FROM lengths WHERE key=?",
                                   (key,)).fetchone()
            if cached is None:
                encoded = encode_supervised(row, tokenizer, 1_000_000)
                full_tokens = len(encoded["input_ids"])
                target_tokens = sum(label != -100 for label in encoded["labels"])
                cache.execute("INSERT INTO lengths VALUES (?, ?, ?)",
                              (key, full_tokens, target_tokens))
            else:
                full_tokens, target_tokens = cached
                cache_hits += 1
            if full_tokens > max_tokens:
                if not allow_skip:
                    raise ValueError(f"sequence exceeds {max_tokens} tokens: {full_tokens}")
                excluded.append({"pack": row["pack"], "sample_id": row["sample_id"],
                                 "view": row["view"],
                                 "reason": f"sequence exceeds {max_tokens} tokens: {full_tokens}"})
            else:
                offsets.append(offset)
                input_tokens += full_tokens
                supervised_tokens += target_tokens
    if not offsets:
        raise ValueError("no training records fit the configured sequence length")
    return offsets, excluded, input_tokens, supervised_tokens, cache_hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = Path(config["train_jsonl"])
    output_dir = Path(config["output_dir"])
    model_dir = Path(config["model_dir"])
    tokenizer_dir = Path(config.get("tokenizer_dir", model_dir))
    cache_path = Path(config.get("length_cache_path",
                                 core.REPO_ROOT / "outputs" / "sft_token_lengths.sqlite3"))
    ignored = (core.REPO_ROOT / "outputs").resolve()
    for path in (source, output_dir, model_dir, tokenizer_dir, cache_path):
        if not path.resolve().is_relative_to(ignored):
            raise ValueError(f"training paths must stay under ignored outputs/: {path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("SFT requires the audited fast tokenizer with offset mappings")
    tokenizer_json_sha = hashlib.sha256(
        (tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest()
    tokenizer_fingerprint = hashlib.sha256(
        (tokenizer_dir / "tokenizer.json").read_bytes() +
        (tokenizer_dir / "tokenizer_config.json").read_bytes()).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path) as cache:
        cache.execute("CREATE TABLE IF NOT EXISTS lengths "
                      "(key TEXT PRIMARY KEY, full_tokens INTEGER, target_tokens INTEGER)")
        offsets, excluded, input_tokens, supervised_tokens, cache_hits = prepare_records(
            source, tokenizer, int(config["max_seq_tokens"]),
            bool(config["allow_overlength_skip"]), cache, tokenizer_fingerprint)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    with source.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            source_digest.update(chunk)
    manifest = {"model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "train_jsonl_sha256": source_digest.hexdigest(),
                "tokenizer_json_sha256": tokenizer_json_sha,
                "tokenizer_fingerprint_sha256": tokenizer_fingerprint,
                "max_seq_tokens": config["max_seq_tokens"],
                "trained_rows": len(offsets), "excluded_rows": excluded,
                "eligible_input_tokens_once": input_tokens,
                "eligible_supervised_tokens_once": supervised_tokens,
                "length_cache_hits": cache_hits,
                "configuration": config}
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trained_rows": len(offsets), "excluded_rows": len(excluded),
                      "manifest": str(output_dir / "input_manifest.json")},
                     ensure_ascii=False))
    if args.check_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; input check completed, training not started")
    for name, expected in config["model_files_sha256"].items():
        digest = hashlib.sha256()
        with (model_dir / name).open("rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"model shard SHA-256 mismatch: {name}")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    quantization = config["quantization"]
    if quantization not in ("none", "4bit"):
        raise ValueError("quantization must be none or 4bit")
    kwargs = {"local_files_only": True, "device_map": {"": torch.cuda.current_device()},
              "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported()
              else torch.float16}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=kwargs["torch_dtype"],
            bnb_4bit_use_double_quant=True)
    if (output_dir / "adapter_final").exists() or list(output_dir.glob("checkpoint-*")):
        raise FileExistsError("training output already contains an adapter or checkpoint")
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.config.use_cache = False
    if quantization == "4bit":
        model = prepare_model_for_kbit_training(model,
                                                use_gradient_checkpointing=True)
    lora = LoraConfig(r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
                      lora_dropout=float(config["lora_dropout"]),
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    seen_indices = set()

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(offsets)

        def __getitem__(self, index):
            seen_indices.add(index)
            with source.open(encoding="utf-8") as input_file:
                input_file.seek(offsets[index])
                row = json.loads(input_file.readline())
            return encode_supervised(row, tokenizer, int(config["max_seq_tokens"]))

    consumed = {"input_tokens": 0, "supervised_tokens": 0}

    def collate(batch):
        consumed["input_tokens"] += sum(len(item["input_ids"]) for item in batch)
        consumed["supervised_tokens"] += sum(
            sum(label != -100 for label in item["labels"]) for item in batch)
        longest = max(len(item["input_ids"]) for item in batch)
        return {key: torch.tensor([
            item[key] + [(-100 if key == "labels" else
                          0 if key == "attention_mask" else tokenizer.pad_token_id)]
            * (longest - len(item[key])) for item in batch], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "labels")}

    training = TrainingArguments(
        output_dir=str(output_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        max_steps=int(config["max_steps"]), learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]), logging_steps=10,
        save_steps=int(config["save_steps"]), save_total_limit=2,
        gradient_checkpointing=True, bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(), report_to=[],
        remove_unused_columns=False, seed=int(config["seed"]),
        dataloader_num_workers=0)
    trainer = Trainer(model=model, args=training, train_dataset=Dataset(),
                      data_collator=collate)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outcome = trainer.train()
    torch.cuda.synchronize()
    training_cost = {"elapsed_seconds": round(time.perf_counter() - start, 2),
                     "unique_training_rows_seen": len(seen_indices),
                     "eligible_training_rows": len(offsets),
                     "consumed_input_tokens": consumed["input_tokens"],
                     "consumed_supervised_tokens": consumed["supervised_tokens"],
                     "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                     "trainer_metrics": outcome.metrics}
    (output_dir / "training_cost.json").write_text(
        json.dumps(training_cost, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    trainer.save_model(str(output_dir / "adapter_final"))
    tokenizer.save_pretrained(str(output_dir / "adapter_final"))


if __name__ == "__main__":
    main()
