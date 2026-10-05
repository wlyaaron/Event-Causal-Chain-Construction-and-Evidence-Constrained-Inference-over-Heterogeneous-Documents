"""OpenAI-compatible research API adapter; credentials never enter output files."""
from __future__ import annotations
import argparse
import getpass
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from task4_core import SYSTEM_PROMPT

DEFAULT_API_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


class TruncatedResponseError(ValueError):
    def __init__(self, usage: dict):
        super().__init__("模型输出被 max_tokens 截断；请增大 --max-tokens")
        self.usage = usage

def completion_url(api_url: str) -> str:
    url = urlsplit(api_url.strip())
    if url.scheme not in {"http", "https"} or not url.netloc:
        raise ValueError("API URL 必须是 http(s) 地址")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("不要在 API URL 中放密钥、查询参数或片段")
    if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("远程 API URL 必须使用 HTTPS")
    path = url.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((url.scheme, url.netloc, path, "", ""))


def read_api_settings(args: argparse.Namespace) -> tuple[str, str, str]:
    api_url = args.api_url or input(f"API URL [{DEFAULT_API_URL}]: ").strip() or DEFAULT_API_URL
    model = args.model or input(f"模型名 [{DEFAULT_MODEL}]: ").strip() or DEFAULT_MODEL
    endpoint = completion_url(api_url)
    key_file = getattr(args, "api_key_file", None)
    if key_file is not None:
        api_key = key_file.read_text(encoding="utf-8-sig").strip()
    elif os.getenv("DEEPSEEK_API_KEY_FILE"):
        api_key = Path(os.environ["DEEPSEEK_API_KEY_FILE"]).read_text(encoding="utf-8-sig").strip()
    elif os.getenv("DEEPSEEK_API_KEY"):
        api_key = os.environ["DEEPSEEK_API_KEY"].strip()
    elif urlsplit(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}:
        api_key = "EMPTY"  # OpenAI-compatible local servers usually ignore this header.
    else:
        api_key = getpass.getpass("DeepSeek API Key（输入不会回显）：").strip()
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()
    if not api_key:
        raise ValueError("API Key 不能为空")
    if not model.strip():
        raise ValueError("模型名不能为空")
    return endpoint, api_key, model.strip()


def http_error_text(exc: HTTPError, api_key: str) -> str:
    detail = exc.read(2048).decode("utf-8", errors="replace").strip()
    detail = detail.replace(api_key, "[REDACTED]")[:500] or "服务器未返回错误正文"
    fields = [f"Content-Type={exc.headers.get('Content-Type', '未知')}"]
    for name in ("Server", "Via", "Content-Length", "x-request-id", "x-ds-request-id", "request-id"):
        value = exc.headers.get(name)
        if value:
            fields.append(f"{name}={value[:120]}")
    return f"API HTTP {exc.code}: {detail}；" + "；".join(fields)


def probe_balance(api_key: str, timeout: int) -> None:
    request = Request("https://api.deepseek.com/user/balance", headers={
        "Authorization": f"Bearer {api_key}", "Accept": "application/json",
    })
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        available = result.get("is_available") if isinstance(result, dict) else None
        print(f"官方 API Key 认证成功；调用余额可用：{available}（不显示金额）。", flush=True)
    except HTTPError as exc:
        raise RuntimeError("认证检查失败；" + http_error_text(exc, api_key)) from exc


def call_model(endpoint: str, api_key: str, model: str, content: str,
               timeout: int = 120, retries: int = 2, max_tokens: int | None = 8192,
               json_mode: bool = True, system_prompt: str | None = SYSTEM_PROMPT,
               return_usage: bool = False) -> str | tuple[str, dict]:
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            choice = result["choices"][0]
            if choice.get("finish_reason") == "length":
                raise TruncatedResponseError(result.get("usage") or {})
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("模型未返回文本内容")
            if return_usage:
                usage = result.get("usage") if isinstance(result, dict) else None
                return text, usage if isinstance(usage, dict) else {}
            return text
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise RuntimeError(http_error_text(exc, api_key)) from exc
        except URLError as exc:
            if attempt == retries:
                raise RuntimeError(f"API 连接失败：{exc.reason}") from exc
        time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")
