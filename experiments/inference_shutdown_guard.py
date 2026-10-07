"""Run the full offline blind inference and stop a dedicated instance afterward.

Exit success requires all 760 predictions to validate. On failure, write status
and logs before shutting down an otherwise idle instance to avoid unattended cost.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import task4_core as core
from experiments.train_shutdown_guard import IdleGate, SHUTDOWN, query_gpu, within_outputs, write_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget-hint", type=int, default=0)
    parser.add_argument("--thinking-budget-tokens", type=int, default=0)
    parser.add_argument("--idle-minutes", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-idle-wait-minutes", type=int, default=120)
    parser.add_argument("--arm-shutdown", action="store_true", required=True)
    args = parser.parse_args()
    if os.geteuid() != 0 or not SHUTDOWN.is_file():
        raise RuntimeError("armed shutdown requires root and /usr/bin/shutdown")
    if min(args.idle_minutes, args.poll_seconds, args.max_idle_wait_minutes) < 1:
        raise ValueError("invalid idle interval")
    output = within_outputs(args.output)
    model_dir = within_outputs(args.model_dir)
    if not model_dir.is_dir() or output.exists():
        raise ValueError("model missing or final output already exists")
    samples = core.discover_samples(core.DEFAULT_DATASET, limit=0)
    if len(samples) != 760 or any(s.track not in "ABC" for s in samples):
        raise ValueError("expected exactly 760 organizer test questions")
    log_file = within_outputs(output.with_name(output.stem + ".guard.log"))
    status_file = within_outputs(output.with_name(output.stem + ".guard.json"))
    if log_file.exists() or status_file.exists():
        raise FileExistsError("guard log/status exists; preserve prior run")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "experiments.run_local_qwen35_thinking",
               "--mode", "blind", "--model-dir", str(model_dir),
               "--output", str(output), "--batch-size", str(args.batch_size),
               "--max-new-tokens", str(args.max_new_tokens),
               "--thinking-budget-hint", str(args.thinking_budget_hint),
               "--thinking-budget-tokens", str(args.thinking_budget_tokens)]
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0"})
    write_status(status_file, "starting", command=command, log_file=str(log_file))
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=core.REPO_ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        write_status(status_file, "running", pid=process.pid,
                     log_file=str(log_file))
        return_code = process.wait()
    try:
        if return_code:
            raise RuntimeError(f"runner exited {return_code}")
        core.validate_file(output, samples, 80000)
        progress = core.read_json(output.with_name(output.name + ".progress.json"))
        if (progress["signature"]["mode"] != "blind"
                or len(progress["records"]) != 760
                or len(progress["details"]) != 760):
            raise ValueError("incomplete progress metadata")
        result = {"success": True, "predictions": 760,
                  "format_fallbacks": sum(item["format_fallback"]
                                          for item in progress["details"])}
        write_status(status_file, "verified_complete", **result)
    except (RuntimeError, ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        result = {"success": False, "reason": str(error), "return_code": return_code}
        write_status(status_file, "run_failed", **result)

    gate = IdleGate(5, 2048, args.idle_minutes * 60)
    deadline = time.monotonic() + args.max_idle_wait_minutes * 60
    while time.monotonic() <= deadline:
        try:
            gpu = query_gpu(0)
            ready, idle_seconds = gate.observe(time.monotonic(), gpu)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            gpu, ready, idle_seconds = None, False, 0
            gate.observe(time.monotonic(), None)
            write_status(status_file, "gpu_check_failed", **result,
                         gpu_error=str(error))
        if ready:
            phase = "shutdown_requested" if result["success"] else "shutdown_requested_after_failure"
            write_status(status_file, phase, **result)
            try:
                subprocess.run(["/bin/sh", str(SHUTDOWN)], check=True, timeout=30)
            except (OSError, subprocess.SubprocessError) as error:
                write_status(status_file, "shutdown_failed", **result,
                             shutdown_error=str(error))
                return 1
            return 0 if result["success"] else 1
        time.sleep(args.poll_seconds)
    write_status(status_file, "idle_timeout_no_shutdown", **result)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
