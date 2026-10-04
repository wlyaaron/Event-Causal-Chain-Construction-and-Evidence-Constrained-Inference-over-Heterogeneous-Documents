"""DeepSeek-compatible research baseline for competition task 4.

Only the supplied documents, questions, and (when present) event/edge files are
sent to the model. Gold answers are never loaded by this program.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from task4_core import (DEFAULT_DATASET, SYSTEM_PROMPT, Sample, read_json, track_of,
                        discover_samples, build_input, parse_prediction, atomic_json,
                        validate_file, PredictionStore)
from task4_api import (DEFAULT_API_URL, DEFAULT_MODEL, completion_url,
                       read_api_settings, http_error_text, probe_balance, call_model)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "deepseek_predictions.json"

def predict(sample: Sample, endpoint: str, api_key: str, model: str,
            timeout: int, retries: int, max_tokens: int,
            max_input_chars: int, json_mode: bool) -> tuple[dict, bool]:
    content, allowed_ids, edges = build_input(sample, max_input_chars)
    for correction in range(2):
        output_cap = max_tokens
        for expansion in range(3):
            try:
                raw = call_model(endpoint, api_key, model, content, timeout, retries,
                                 output_cap, json_mode=json_mode)
                break
            except ValueError as exc:
                if "max_tokens 截断" not in str(exc) or expansion == 2:
                    raise
                output_cap = min(output_cap * 2, 32768)
        try:
            record = parse_prediction(raw, sample, allowed_ids)
            chain = record["evidence_chain"]
            unsupported = bool(edges and any((a, b) not in edges for a, b in zip(chain, chain[1:])))
            return record, unsupported
        except ValueError as exc:
            if correction:
                if sample.question_type == "unanswerable":
                    fallback = '{"answer":"无法确定","evidence_chain":[],"confidence":null}'
                    return parse_prediction(fallback, sample, allowed_ids), False
                raise
            content += ("\n\n上一版输出未通过提交格式校验：" + str(exc)
                        + "。请仅输出修正后的 JSON；证据链只用上述真实事件 ID。"
                          "无法确定且没有可核查证据时，使用精确拒答格式。")
    raise AssertionError("unreachable")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="赛题4 DeepSeek API 实验基线；默认只试运行 3 题")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="测试集根目录或单个文档包")
    parser.add_argument("--track", choices=["all", "A", "B", "C", "train"], default="all")
    parser.add_argument("--limit", type=int, default=3, help="最多处理的问题数；0 表示全部，默认 3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default=os.getenv("DEEPSEEK_API_URL"), help="API 基础地址或完整 /chat/completions 地址")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL"), help="服务商提供的准确模型名")
    parser.add_argument("--api-key-file", type=Path, help="从本地 UTF-8 文本文件读取 Key；文件内容不会显示或写入结果")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="单次最大生成 token 数，含模型思考；默认 8192")
    parser.add_argument("--max-input-chars", type=int, default=80000)
    parser.add_argument("--workers", type=int, default=1, help="并发请求数，1 到 8；结果仍按问题顺序保存")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和输入规模，不调用 API")
    parser.add_argument("--probe-api", action="store_true", help="用两条短消息依次测试基础调用与 JSON 模式，不发送数据集")
    parser.add_argument("--no-json-mode", action="store_true", help="不向 API 发送 response_format，仍要求模型输出 JSON")
    parser.add_argument("--validate", type=Path, help="只校验已有 JSON 的覆盖率和格式")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有完整或未完成的输出")
    args = parser.parse_args(argv)

    try:
        if args.probe_api:
            endpoint, api_key, model = read_api_settings(args)
            print(f"诊断目标：{endpoint}，模型：{model}；不会发送数据集或显示 Key。")
            if urlsplit(endpoint).hostname == "api.deepseek.com":
                print("[认证] 检查官方 API Key 状态……", flush=True)
                probe_balance(api_key, args.timeout)
            print("[1/2] 测试极简聊天请求……", flush=True)
            call_model(endpoint, api_key, model, "Hello", args.timeout, 0, None,
                       json_mode=False, system_prompt=None)
            print("极简请求成功。", flush=True)
            print("[2/2] 测试 JSON 输出模式……", flush=True)
            call_model(endpoint, api_key, model, "请输出包含 ok=true 的 JSON 对象。",
                       args.timeout, 0, None, json_mode=True,
                       system_prompt='请只输出 JSON 对象，例如 {"ok": true}。')
            print("JSON 模式成功。当前文件中的 Key 和接口均可用。")
            return 0

        samples = discover_samples(args.dataset, args.track, args.limit)
        counts: dict[str, int] = {}
        for sample in samples:
            counts[sample.track] = counts.get(sample.track, 0) + 1
        print(f"选中 {len(samples)} 题：{counts}")
        if args.validate:
            count = validate_file(args.validate, samples, args.max_input_chars)
            print(f"格式和覆盖率校验通过：{count} 题")
            return 0
        if args.dry_run:
            for sample in samples[:5]:
                content, allowed_ids, edges = build_input(sample, args.max_input_chars)
                print(f"{sample.sample_id}: {len(content)} 字符，{len(allowed_ids)} 个候选 ID，{len(edges)} 条已给因果边")
            print("未发出 API 请求；未读取 gold 答案。")
            return 0

        if not 1 <= args.workers <= 8:
            raise ValueError("--workers 必须在 1 到 8 之间")

        endpoint, api_key, model = read_api_settings(args)
        signature = hashlib.sha256("\n".join(s.sample_id for s in samples).encode()).hexdigest()
        metadata = {"dataset": str(args.dataset.resolve()), "track": args.track,
                    "limit": args.limit, "model": model, "endpoint": endpoint,
                    "sample_ids_sha256": signature}
        if args.no_json_mode:
            metadata["json_mode"] = "disabled"
        store = PredictionStore(args.output, samples, metadata,
                                args.max_input_chars, args.overwrite)
        if store.completed:
            print(f"从 {len(store.completed)} 条已保存结果继续")

        def work(sample: Sample) -> tuple[dict, bool]:
            return predict(sample, endpoint, api_key, model, args.timeout,
                           args.retries, args.max_tokens, args.max_input_chars,
                           not args.no_json_mode)

        start = len(store.completed)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            next_submit = start
            for index in range(start, len(samples)):
                while next_submit < min(len(samples), index + args.workers):
                    pending[next_submit] = pool.submit(work, samples[next_submit])
                    next_submit += 1
                record, unsupported = pending.pop(index).result()
                sample = samples[index]
                if unsupported:
                    print(f"提示：{sample.sample_id} 的证据链含未在给定图中直接列出的相邻边，请人工复核。", file=sys.stderr)
                store.append(record)
                print(f"[{index + 1}/{len(samples)}] {sample.sample_id} 已保存", flush=True)

        store.finish()
        print(f"完成：{store.output}。官方赛事禁止闭源 API；正式提交须使用符合规则的离线开源模型，并复核答案和证据链。")
        return 0
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
