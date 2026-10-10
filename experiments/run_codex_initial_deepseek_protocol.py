"""Replay the earliest DeepSeek visible protocol through Codex on frozen validation.

This is a Codex transport comparison, not a byte-identical Chat Completions
request: Codex accepts one prompt and supplies its own runtime instructions.
Gold files are never opened here. Invalid outputs are preserved, not repaired.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path

import task4_core as core
import task4_workflow as workflow

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "baseline_deepseek_2026-10-04" / "source_initial_1990c861.zip"
MANIFEST = ROOT / "experiments" / "validation_diagnostic_48_20261008.json"
CODEX = Path(r"C:\Users\Aaron\AppData\Local\OpenAI\Codex\bin\9691020b546a15b2\codex.exe")
SYSTEM_SHA256 = "fa86a00fd3475e1f2ae7179c0d08ad14dca7fd8182ad356435bacf22468588fe"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def historical_module() -> types.ModuleType:
    with zipfile.ZipFile(ARCHIVE) as archive:
        source = archive.read("run_deepseek.py").decode("utf-8")
    module = types.ModuleType("task4_original_deepseek_1990c861")
    module.__file__ = str(ARCHIVE)
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    if sha(module.SYSTEM_PROMPT) != SYSTEM_SHA256:
        raise ValueError("Historical SYSTEM_PROMPT changed")
    return module


def stage_view(sample: core.Sample, view: str, stage_root: Path, old: types.ModuleType):
    """Stage only public view files so the historical build_input runs unchanged."""
    content, allowed, _ = workflow.training_view_input(sample, view)
    payload = json.loads(content)
    stage = stage_root / f"测试集{view}_validation" / sample.pack.name
    stage.mkdir(parents=True, exist_ok=True)
    for document in payload["documents"]:
        (stage / f'{document["doc_id"]}.txt').write_text(document["text"], encoding="utf-8")
    if payload["events"] is not None:
        (stage / "事件列表.json").write_text(json.dumps(payload["events"], ensure_ascii=False), encoding="utf-8")
    if payload["causal_relations"] is not None:
        (stage / "事件因果关系列表.json").write_text(json.dumps(payload["causal_relations"], ensure_ascii=False), encoding="utf-8")
    original_sample = old.Sample(stage, view, sample.sample_id, sample.question_type, sample.question)
    historical_content, historical_allowed, _ = old.build_input(original_sample, 80000)
    if historical_content != content or historical_allowed != allowed:
        raise ValueError(f"Historical input differs from validation view: {sample.sample_id}/{view}")
    return original_sample, historical_content, allowed


def call_codex(model: str, prompt: str, workdir: Path, timeout: int) -> dict:
    command = [str(CODEX), "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
               "--skip-git-repo-check", "-s", "read-only", "-C", str(workdir),
               "-m", model, "--json", "-"]
    start = time.perf_counter()
    try:
        result = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout,
                                cwd=workdir, check=False)
    except subprocess.TimeoutExpired:
        return {"elapsed_seconds": round(time.perf_counter()-start, 3),
                "returncode": None, "raw": None, "usage": None,
                "tool_events": [], "error": "timeout"}
    events = []
    for line in result.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    messages = [event["item"]["text"] for event in events
                if event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "agent_message"]
    completions = [event for event in events if event.get("type") == "turn.completed"]
    tools = [event.get("item", {}).get("type") for event in events
             if event.get("type") == "item.started"
             and event.get("item", {}).get("type") not in {"agent_message", "reasoning"}]
    return {"elapsed_seconds": round(time.perf_counter()-start, 3),
            "returncode": result.returncode, "raw": messages[-1] if messages else None,
            "usage": completions[-1].get("usage") if completions else None,
            "tool_events": tools, "stderr_tail": result.stderr[-1000:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--model", default="gpt-6-sol")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true",
                        help="Resend the identical request for saved invalid outputs; no correction text")
    args = parser.parse_args()
    if not 1 <= args.limit <= 48 or not CODEX.is_file():
        raise ValueError("Invalid limit or missing Codex CLI")
    old = historical_module()
    manifest_bytes = MANIFEST.read_bytes()
    rows = json.loads(manifest_bytes)["rows"]
    if len(rows) != 48:
        raise ValueError("Frozen manifest changed")
    samples = {s.sample_id: s for s in core.discover_samples(ROOT / "数据集" / "训练集", limit=0)
               if s.pack.name in {r["pack"] for r in rows}}
    output_root = ROOT / "outputs" / "gpt6sol_original_protocol_48"
    stage_root = output_root / "staged_views"
    attempts_path = output_root / "attempts.jsonl"
    output_path = output_root / "predictions.json"
    meta_path = output_root / "metadata.json"
    blind_workdir = output_root / "blind_workdir"
    signature = {"manifest_sha256": sha(manifest_bytes), "historical_source_sha256": sha(ARCHIVE.read_bytes()),
                 "system_prompt_sha256": sha(old.SYSTEM_PROMPT), "model": args.model,
                 "transport": "codex exec single prompt; historical system + two newlines + historical user JSON",
                 "output_schema": None, "format_retries": 0, "gold_sent": False}
    if meta_path.exists() and core.read_json(meta_path) != signature:
        raise ValueError("Resume metadata mismatch")
    done = {}
    if attempts_path.exists():
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["sample_id"], row["view"])] = row
    prepared = []
    for row in rows[:args.limit]:
        sample = samples[row["sample_id"]]
        if sample.pack.name != row["pack"]:
            raise ValueError("Manifest pack mismatch")
        historical_sample, content, allowed = stage_view(sample, row["view"], stage_root, old)
        prepared.append((row, historical_sample, content, allowed))
    if args.dry_run:
        print(json.dumps({"items": len(prepared), "total_user_chars": sum(len(p[2]) for p in prepared),
                          "system_prompt_sha256": sha(old.SYSTEM_PROMPT), "gold_sent": False},
                         ensure_ascii=False))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    blind_workdir.mkdir(exist_ok=True)
    if not meta_path.exists():
        core.atomic_json(meta_path, signature)
    for index, (row, sample, content, allowed) in enumerate(prepared, 1):
        key = (row["sample_id"], row["view"])
        if key in done and (done[key]["parsed"] is not None or not args.retry_invalid):
            continue
        # The visible prompt is exact historical text; Codex cannot represent
        # its separate system/user roles or JSON mode through `codex exec`.
        prompt = old.SYSTEM_PROMPT + "\n\n" + content
        attempt = call_codex(args.model, prompt, blind_workdir, args.timeout)
        parsed = None
        try:
            if attempt["tool_events"]:
                raise ValueError("Codex used tools")
            if attempt["returncode"] != 0 or attempt["raw"] is None:
                raise ValueError("Codex returned no completed answer")
            parsed = old.parse_prediction(attempt["raw"], sample, allowed)
        except ValueError as exc:
            attempt["validation_error"] = str(exc)
        record = {"sample_id": sample.sample_id, "view": row["view"],
                  "input_sha256": sha(content), "prompt_sha256": sha(prompt),
                  "input_chars": len(content), "parsed": parsed, "attempt": attempt}
        with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        done[key] = record
        print(json.dumps({"index": index, "sample_id": sample.sample_id,
                          "view": row["view"], "valid": parsed is not None,
                          "elapsed_seconds": attempt["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    predictions = [{"view": row["view"], **done[(row["sample_id"], row["view"])]["parsed"]}
                   for row in rows[:args.limit] if (row["sample_id"], row["view"]) in done
                   and done[(row["sample_id"], row["view"])]["parsed"] is not None]
    core.atomic_json(output_path, predictions)
    print(json.dumps({"valid": len(predictions), "attempted": len(done),
                      "target": args.limit, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
