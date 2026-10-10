"""Blind DeepSeek API replay of the earliest archived protocol on frozen validation.

Only public view material is sent. Gold is never opened by this module.
The HTTP body follows the first archived run_deepseek.py request layout. The
max_tokens cap is explicit because the original 1200 cap can truncate current
thinking-mode outputs; an 8192 cap also appears in the archived run-start code.
Raw failures are retained; no correction prompt is used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import task4_core as core
from experiments.run_codex_initial_deepseek_protocol import (
    ARCHIVE, MANIFEST, ROOT, historical_module, sha, stage_view,
)

KEY_FILE = Path(r"C:\Users\Aaron\Desktop\新建 文本文档.txt")
ENDPOINT = "https://api.deepseek.com/chat/completions"
OUT_1200 = ROOT / "outputs" / "deepseek_initial_protocol_48"


def read_key() -> str:
    key = KEY_FILE.read_text(encoding="utf-8-sig").strip()
    if key.startswith("DEEPSEEK_API_KEY="):
        key = key.split("=", 1)[1].strip().strip('"\'')
    if not key.startswith("sk-") or any(ch.isspace() for ch in key):
        raise ValueError("Key file does not contain one DeepSeek API key")
    return key


def call_api(key: str, model: str, system: str, content: str,
             timeout: int = 120, retries: int = 2, max_tokens: int = 1200) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }, ensure_ascii=False).encode("utf-8")
    request = Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    start = time.perf_counter()
    for retry in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            choice = result["choices"][0]
            raw = choice["message"]["content"]
            return {"elapsed_seconds": round(time.perf_counter()-start, 3),
                    "request_body_sha256": sha(body), "raw": raw if isinstance(raw, str) else None,
                    "response_model": result.get("model"), "usage": result.get("usage"),
                    "finish_reason": choice.get("finish_reason"),
                    "system_fingerprint": result.get("system_fingerprint"), "http_retries": retry}
        except HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace").replace(key, "[REDACTED]")
            if exc.code not in {429, 500, 502, 503, 504} or retry == retries:
                return {"elapsed_seconds": round(time.perf_counter()-start, 3),
                        "request_body_sha256": sha(body), "raw": None,
                        "http_status": exc.code, "error": detail, "http_retries": retry}
        except URLError as exc:
            if retry == retries:
                return {"elapsed_seconds": round(time.perf_counter()-start, 3),
                        "request_body_sha256": sha(body), "raw": None,
                        "error": str(exc.reason).replace(key, "[REDACTED]"),
                        "http_retries": retry}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return {"elapsed_seconds": round(time.perf_counter()-start, 3),
                    "request_body_sha256": sha(body), "raw": None,
                    "error": str(exc).replace(key, "[REDACTED]"), "http_retries": retry}
        time.sleep(min(2**retry, 8))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--retry-invalid", action="store_true",
                        help="Resend the same original request for invalid records")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 48 or args.max_tokens < 1:
        raise ValueError("Invalid limit")
    old = historical_module()
    manifest_bytes = MANIFEST.read_bytes()
    rows = json.loads(manifest_bytes)["rows"]
    if len(rows) != 48:
        raise ValueError("Frozen manifest changed")
    samples = {s.sample_id: s for s in core.discover_samples(ROOT / "数据集" / "训练集", limit=0)
               if s.pack.name in {r["pack"] for r in rows}}
    out = OUT_1200 if args.max_tokens == 1200 else ROOT / "outputs" / f"deepseek_initial_protocol_48_max{args.max_tokens}"
    stage_root = out / "staged_views"
    attempts_path = out / "attempts.jsonl"
    output_path = out / "predictions.json"
    meta_path = out / "metadata.json"
    signature = {"manifest_sha256": sha(manifest_bytes),
                 "historical_source_sha256": sha(ARCHIVE.read_bytes()),
                 "system_prompt_sha256": sha(old.SYSTEM_PROMPT),
                 "model_alias": args.model, "endpoint": ENDPOINT,
                 "response_format": {"type": "json_object"}, "max_tokens": args.max_tokens,
                 "gold_sent": False}
    if meta_path.exists() and core.read_json(meta_path) != signature:
        raise ValueError("Resume metadata mismatch")
    done = {}
    if attempts_path.exists():
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[(record["sample_id"], record["view"])] = record
    prepared = []
    for row in rows[:args.limit]:
        sample = samples[row["sample_id"]]
        if sample.pack.name != row["pack"]:
            raise ValueError("Manifest pack mismatch")
        historical_sample, content, allowed = stage_view(sample, row["view"], stage_root, old)
        prepared.append((row, historical_sample, content, allowed))
    if args.dry_run:
        print(json.dumps({"items": len(prepared), "total_user_chars": sum(len(x[2]) for x in prepared),
                          "model_alias": args.model, "gold_sent": False}, ensure_ascii=False))
        return
    key = read_key()
    out.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        core.atomic_json(meta_path, signature)
    for index, (row, sample, content, allowed) in enumerate(prepared, 1):
        identity = (row["sample_id"], row["view"])
        if identity in done and (done[identity]["parsed"] is not None or not args.retry_invalid):
            continue
        attempt = call_api(key, args.model, old.SYSTEM_PROMPT, content, args.timeout,
                           max_tokens=args.max_tokens)
        parsed = None
        try:
            if attempt["raw"] is None:
                raise ValueError(attempt.get("error") or "No model output")
            if attempt.get("finish_reason") == "length":
                raise ValueError(f"Model output truncated by max_tokens={args.max_tokens}")
            parsed = old.parse_prediction(attempt["raw"], sample, allowed)
        except ValueError as exc:
            attempt["validation_error"] = str(exc).replace(key, "[REDACTED]")
        record = {"sample_id": sample.sample_id, "view": row["view"],
                  "input_sha256": sha(content), "parsed": parsed, "attempt": attempt}
        with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        done[identity] = record
        print(json.dumps({"index": index, "sample_id": sample.sample_id, "view": row["view"],
                          "valid": parsed is not None, "elapsed_seconds": attempt["elapsed_seconds"],
                          "response_model": attempt.get("response_model"),
                          "http_status": attempt.get("http_status")}, ensure_ascii=False), flush=True)
        if attempt.get("http_status") in {401, 402, 403}:
            raise RuntimeError("Authentication or account billing rejected; stopping before further calls")
    predictions = [{"view": row["view"], **done[(row["sample_id"], row["view"])]["parsed"]}
                   for row in rows[:args.limit] if (row["sample_id"], row["view"]) in done
                   and done[(row["sample_id"], row["view"])]["parsed"] is not None]
    core.atomic_json(output_path, predictions)
    print(json.dumps({"valid": len(predictions), "attempted": len(done),
                      "target": args.limit, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
