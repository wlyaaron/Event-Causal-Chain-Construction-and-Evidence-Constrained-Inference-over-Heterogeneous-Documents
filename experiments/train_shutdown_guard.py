"""Run offline LLaMA-Factory SFT and shut down only after verified completion.

GPU idleness is an additional guard, never proof that training finished. Use
--arm-shutdown only on a dedicated AutoDL instance. All logs stay in outputs/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = (REPO_ROOT / "outputs").resolve()
DEFAULT_TRAINER = Path("/root/autodl-tmp/task4-qwen35-venv/bin/llamafactory-cli")
SHUTDOWN = Path("/usr/bin/shutdown")


def within_outputs(path: Path) -> Path:
    resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(OUTPUTS_ROOT):
        raise ValueError(f"path must stay in ignored outputs/: {path}")
    return resolved


def output_dir_from_config(config: Path) -> Path:
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("output_dir"), str):
        raise ValueError("YAML config has no output_dir")
    output = Path(value["output_dir"])
    return within_outputs(output if output.is_absolute() else REPO_ROOT / output)


def validate_artifacts(output_dir: Path, min_steps: int = 1) -> dict:
    """Reject a zero-exit run with missing, stale, or incomplete final output."""
    required = ("adapter_config.json", "adapter_model.safetensors",
                "trainer_state.json", "train_results.json")
    for name in required:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size < (1024 if name.endswith(".safetensors") else 2):
            raise ValueError(f"missing or empty final artifact: {path}")
    adapter = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    state = json.loads((output_dir / "trainer_state.json").read_text(encoding="utf-8"))
    results = json.loads((output_dir / "train_results.json").read_text(encoding="utf-8"))
    step, maximum = state.get("global_step"), state.get("max_steps")
    if not isinstance(adapter, dict) or not isinstance(step, int) or not isinstance(maximum, int):
        raise ValueError("invalid adapter or trainer state")
    if maximum < min_steps or step != maximum:
        raise ValueError(f"training stopped early: global_step={step}, max_steps={maximum}")
    runtime, loss = results.get("train_runtime"), results.get("train_loss")
    if not all(isinstance(number, (int, float)) and math.isfinite(number)
               for number in (runtime, loss)) or runtime <= 0:
        raise ValueError("invalid final training metrics")
    return {"global_step": step, "max_steps": maximum,
            "train_runtime_seconds": runtime, "train_loss": loss,
            "adapter_bytes": (output_dir / "adapter_model.safetensors").stat().st_size}


@dataclass(frozen=True)
class GpuSample:
    utilization_pct: int
    memory_mib: int
    compute_processes: tuple[str, ...]


def query_gpu(index: int) -> GpuSample:
    common = ["nvidia-smi", "-i", str(index)]
    usage = subprocess.run(
        common + ["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip().splitlines()
    if len(usage) != 1:
        raise ValueError(f"expected one GPU usage row, got {len(usage)}")
    fields = [item.strip() for item in usage[0].split(",")]
    if len(fields) != 2:
        raise ValueError(f"unexpected nvidia-smi usage row: {usage[0]}")
    apps = subprocess.run(
        common + ["--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    return GpuSample(int(fields[0]), int(fields[1]), tuple(apps.splitlines()) if apps else ())


class IdleGate:
    def __init__(self, utilization_limit: int, memory_limit_mib: int, idle_seconds: int):
        self.utilization_limit = utilization_limit
        self.memory_limit_mib = memory_limit_mib
        self.idle_seconds = idle_seconds
        self.since: float | None = None

    def observe(self, now: float, sample: GpuSample | None) -> tuple[bool, float]:
        idle = (sample is not None and sample.utilization_pct <= self.utilization_limit
                and sample.memory_mib <= self.memory_limit_mib
                and not sample.compute_processes)
        if not idle:
            self.since = None
            return False, 0.0
        if self.since is None:
            self.since = now
        elapsed = now - self.since
        return elapsed >= self.idle_seconds, elapsed


def write_status(path: Path, phase: str, **details: object) -> None:
    body = {"phase": phase, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(), **details}
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(body, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--utilization-limit", type=int, default=5)
    parser.add_argument("--memory-limit-mib", type=int, default=2048)
    parser.add_argument("--idle-minutes", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-idle-wait-minutes", type=int, default=120)
    parser.add_argument("--min-steps", type=int, default=1000,
                        help="minimum completed optimizer steps required before armed shutdown")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm-shutdown", action="store_true")
    mode.add_argument("--no-shutdown", action="store_true")
    mode.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.utilization_limit <= 100 or args.memory_limit_mib < 0:
        parser.error("invalid GPU idle threshold")
    if min(args.idle_minutes, args.poll_seconds, args.max_idle_wait_minutes, args.min_steps) < 1:
        parser.error("idle, poll, and maximum wait periods must be positive")
    return args


def main() -> int:
    args = parse_args()
    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    output_dir = output_dir_from_config(config)
    if args.check_existing:
        print(json.dumps({"artifacts": validate_artifacts(output_dir),
                          "gpu": query_gpu(args.gpu_index).__dict__}, ensure_ascii=False))
        return 0

    default_prefix = Path("outputs/llamafactory_logs") / config.stem
    log_file = within_outputs(args.log_file or Path(str(default_prefix) + ".train.log"))
    status_file = within_outputs(args.status_file or Path(str(default_prefix) + ".watch.json"))
    if (log_file == status_file or output_dir in log_file.parents or output_dir in status_file.parents
            or log_file.exists() or status_file.exists() or output_dir.exists()):
        raise FileExistsError("output, log, or status path already exists; keep old run for audit")
    trainer = args.trainer.resolve()
    if not trainer.is_file() or not os.access(trainer, os.X_OK):
        raise FileNotFoundError(trainer)
    if args.arm_shutdown and (os.geteuid() != 0 or not SHUTDOWN.is_file()):
        raise RuntimeError("armed shutdown requires root and the platform shutdown command")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
                        "HF_HOME": "/root/autodl-tmp/task4-hf-cache"})
    write_status(status_file, "starting", config=str(config), output_dir=str(output_dir),
                 log_file=str(log_file), armed=args.arm_shutdown)
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(trainer), "train", str(config)], cwd=REPO_ROOT,
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            write_status(status_file, "training", pid=process.pid)
            time.sleep(args.poll_seconds)
        return_code = process.returncode
    if return_code != 0:
        write_status(status_file, "training_failed", return_code=return_code,
                     log_file=str(log_file))
        return 1
    try:
        artifacts = validate_artifacts(output_dir, args.min_steps if args.arm_shutdown else 1)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        write_status(status_file, "artifacts_failed", reason=str(error))
        return 1

    gate = IdleGate(args.utilization_limit, args.memory_limit_mib, args.idle_minutes * 60)
    deadline = time.monotonic() + args.max_idle_wait_minutes * 60
    while time.monotonic() <= deadline:
        try:
            sample = query_gpu(args.gpu_index)
            query_error = None
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            sample, query_error = None, str(error)
        ready, idle_for = gate.observe(time.monotonic(), sample)
        write_status(status_file, "waiting_for_idle", artifacts=artifacts,
                     gpu=sample.__dict__ if sample else None, gpu_error=query_error,
                     idle_seconds=round(idle_for), required_idle_seconds=args.idle_minutes * 60)
        if ready:
            if not args.arm_shutdown:
                write_status(status_file, "verified_no_shutdown", artifacts=artifacts)
                return 0
            write_status(status_file, "shutdown_requested", artifacts=artifacts)
            try:
                # This AutoDL image supplies /usr/bin/shutdown as a shell script without a shebang.
                subprocess.run(["/bin/sh", str(SHUTDOWN)], check=True, timeout=30)
            except (OSError, subprocess.SubprocessError) as error:
                write_status(status_file, "shutdown_failed", reason=str(error), artifacts=artifacts)
                return 1
            return 0
        time.sleep(args.poll_seconds)
    write_status(status_file, "idle_timeout_no_shutdown", artifacts=artifacts)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
