"""Record CPU/GPU and dependency facts before a Task 4 offline SFT run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

import task4_core as core


def _memory_limit() -> int | None:
    for path in (Path("/sys/fs/cgroup/memory.max"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value.isdigit():
                return int(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ignored = (core.REPO_ROOT / "outputs").resolve()
    if not args.output.resolve().is_relative_to(ignored):
        raise ValueError("environment report must stay under ignored outputs/")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_dir = Path(config["model_dir"])
    train_file = Path(config["train_jsonl"])
    if not model_dir.resolve().is_relative_to(ignored) or not train_file.resolve().is_relative_to(ignored):
        raise ValueError("model and training data must stay under ignored outputs/")
    import torch

    versions = {}
    for package in ("torch", "transformers", "tokenizers", "huggingface-hub",
                    "peft", "bitsandbytes", "accelerate", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu.append({"name": properties.name,
                    "total_memory_bytes": properties.total_memory})
    disk = shutil.disk_usage(model_dir if model_dir.exists() else ignored)
    report = {"gpu": gpu, "cuda_available": torch.cuda.is_available(),
              "cgroup_memory_limit_bytes": _memory_limit(),
              "disk_free_bytes": disk.free,
              "model_id": config["model_id"],
              "model_revision": config["model_revision"],
              "model_files_present": {name: (model_dir / name).is_file()
                                      for name in config["model_files_sha256"]},
              "training_data_present": train_file.is_file(),
              "versions": versions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    core.atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
