"""DeepSeek-compatible research baseline for competition task 4.

Only the supplied documents, questions, and (when present) event/edge files are
sent to the model. Gold answers are never loaded by this program.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "数据集" / "抽样测试集_100"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "deepseek_predictions.json"
DEFAULT_API_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"

SYSTEM_PROMPT = """你是赛题4的证据约束问答系统。只依据用户消息给出的文档、事件和因果关系回答；不得使用外部知识补齐材料缺口。
请输出一个 JSON 对象，且仅包含 answer、evidence_chain、confidence 三个字段。
answer：简洁回答问题，覆盖有材料支持的关键事实；若材料不足、互相冲突或问题前提无法确认，严格写“无法确定”。反事实问题只作有边界推断，不把可能性说成确定事实。
evidence_chain：按原因到结果顺序排列的事件 ID。给定事件列表时使用其中的 event_id；未给定事件列表时，现有数据每篇文档对应一个候选事件，使用相应的 D001 等文档 ID。不要添加材料中不存在的 ID，不要把时间先后自动当作因果。
confidence：可回答时给出 0 到 1 的数值；回答“无法确定”时为 null，证据链必须为空数组。
任务 A 若给出因果边，优先使用有向边支持的路径；任务 B/C 没有边时，需要从文档判断因果方向。所有结论都须受材料约束。"""


@dataclass(frozen=True)
class Sample:
    pack: Path
    track: str
    sample_id: str
    question_type: str
    question: str


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def track_of(pack: Path) -> str:
    for part in pack.parts:
        if part.startswith("测试集A_"):
            return "A"
        if part.startswith("测试集B_"):
            return "B"
        if part.startswith("测试集C_"):
            return "C"
    return "train"


def discover_samples(dataset: Path, track: str = "all", limit: int = 3) -> list[Sample]:
    if not dataset.is_dir():
        raise ValueError(f"数据目录不存在：{dataset}")
    if limit < 0:
        raise ValueError("--limit 不能小于 0；使用 0 表示全部问题")
    samples: list[Sample] = []
    seen: set[str] = set()
    for question_file in sorted(dataset.rglob("问题.json")):
        pack = question_file.parent
        current_track = track_of(pack)
        if track != "all" and current_track != track:
            continue
        questions = read_json(question_file)
        if not isinstance(questions, list):
            raise ValueError(f"问题文件不是数组：{question_file}")
        for item in questions:
            if not isinstance(item, dict):
                raise ValueError(f"问题格式错误：{question_file}")
            sample_id = item.get("sample_id")
            question_type = item.get("question_type")
            question = item.get("question")
            if not all(isinstance(value, str) and value.strip() for value in
                       (sample_id, question_type, question)):
                raise ValueError(f"问题缺少必要字段：{question_file}")
            if sample_id in seen:
                raise ValueError(f"重复的 sample_id：{sample_id}")
            seen.add(sample_id)
            samples.append(Sample(pack, current_track, sample_id,
                                  question_type, question))
    if not samples:
        raise ValueError(f"没有找到符合条件的问题：{dataset}，档位 {track}")
    return samples if limit == 0 else samples[:limit]


def build_input(sample: Sample, max_input_chars: int = 80000) -> tuple[str, set[str], set[tuple[str, str]]]:
    document_files = sorted(sample.pack.glob("D[0-9]*.txt"))
    if not document_files:
        raise ValueError(f"文档包缺少 D*.txt：{sample.pack}")
    documents = [{"doc_id": path.stem, "text": path.read_text(encoding="utf-8")}
                 for path in document_files]
    event_file = sample.pack / "事件列表.json"
    edge_file = sample.pack / "事件因果关系列表.json"
    events = read_json(event_file) if event_file.exists() else None
    edges = read_json(edge_file) if edge_file.exists() else None
    if events is not None and not isinstance(events, list):
        raise ValueError(f"事件列表不是数组：{event_file}")
    if edges is not None and not isinstance(edges, list):
        raise ValueError(f"因果关系列表不是数组：{edge_file}")

    allowed_ids = ({event["event_id"] for event in events} if events is not None
                   else {document["doc_id"] for document in documents})
    directed_edges: set[tuple[str, str]] = set()
    if edges:
        for edge in edges:
            cause = edge.get("cause_event_id")
            result = edge.get("result_event_id", edge.get("effect_event_id"))
            if isinstance(cause, str) and isinstance(result, str):
                directed_edges.add((cause, result))

    # In particular, never read or include gold/问答对_答案.json here.
    prompt = {
        "track": sample.track,
        "sample_id": sample.sample_id,
        "question_type": sample.question_type,
        "question": sample.question,
        "documents": documents,
        "events": events,
        "causal_relations": edges,
    }
    content = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
    if len(content) > max_input_chars:
        raise ValueError(f"{sample.sample_id} 输入为 {len(content)} 字符，超过 --max-input-chars={max_input_chars}；程序不会静默截断材料")
    return content, allowed_ids, directed_edges


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


def call_model(endpoint: str, api_key: str, model: str, content: str,
               timeout: int = 120, retries: int = 2, max_tokens: int = 1200) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }, ensure_ascii=False).encode("utf-8")
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
                raise ValueError("模型输出被 max_tokens 截断；请增大 --max-tokens")
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("模型未返回文本内容")
            return text
        except HTTPError as exc:
            detail = exc.read(400).decode("utf-8", errors="replace").replace(api_key, "[REDACTED]")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            if attempt == retries:
                raise RuntimeError(f"API 连接失败：{exc.reason}") from exc
        time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def parse_prediction(raw: str, sample: Sample, allowed_ids: set[str]) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sample.sample_id} 模型返回的不是完整 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{sample.sample_id} 模型返回的不是 JSON 对象")
    answer = value.get("answer")
    chain = value.get("evidence_chain")
    confidence = value.get("confidence")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{sample.sample_id} answer 不能为空")
    answer = answer.strip()
    if not isinstance(chain, list) or any(not isinstance(x, str) for x in chain):
        raise ValueError(f"{sample.sample_id} evidence_chain 必须是 ID 字符串数组")
    chain = [item.strip() for item in chain]
    if any(item not in allowed_ids for item in chain):
        unknown = sorted(set(chain) - allowed_ids)
        raise ValueError(f"{sample.sample_id} 证据 ID 不在材料中：{unknown}")
    if len(chain) != len(set(chain)):
        raise ValueError(f"{sample.sample_id} 证据链含重复 ID")
    if answer == "无法确定":
        if chain or confidence is not None:
            raise ValueError(f"{sample.sample_id} 拒答时证据链必须为空、置信度必须为 null")
    else:
        if not chain:
            raise ValueError(f"{sample.sample_id} 可回答题必须提供证据链")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"{sample.sample_id} confidence 必须是 0 到 1 的数值")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"{sample.sample_id} confidence 超出 0 到 1")
        confidence = float(confidence)
    return {
        "sample_id": sample.sample_id,
        "answer": answer,
        "evidence_chain": chain,
        "confidence": confidence,
        "question_type": sample.question_type,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_file(path: Path, samples: list[Sample], max_input_chars: int) -> int:
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError("提交文件必须是 JSON 数组")
    sample_map = {sample.sample_id: sample for sample in samples}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("sample_id") not in sample_map:
            raise ValueError("提交文件包含未知 sample_id 或无效对象")
        sample = sample_map[record["sample_id"]]
        if sample.sample_id in seen:
            raise ValueError(f"提交文件包含重复 sample_id：{sample.sample_id}")
        seen.add(sample.sample_id)
        _, allowed_ids, _ = build_input(sample, max_input_chars)
        expected = parse_prediction(json.dumps(record, ensure_ascii=False), sample, allowed_ids)
        if record != expected:
            raise ValueError(f"{sample.sample_id} 字段不符合提交格式，或 question_type 与原题不符")
    missing = set(sample_map) - seen
    if missing:
        raise ValueError(f"提交文件缺少 {len(missing)} 个问题；例如 {sorted(missing)[0]}")
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="赛题4 DeepSeek API 实验基线；默认只试运行 3 题")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="测试集根目录或单个文档包")
    parser.add_argument("--track", choices=["all", "A", "B", "C", "train"], default="all")
    parser.add_argument("--limit", type=int, default=3, help="最多处理的问题数；0 表示全部，默认 3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default=os.getenv("DEEPSEEK_API_URL"), help="API 基础地址或完整 /chat/completions 地址")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL"), help="服务商提供的准确模型名")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-input-chars", type=int, default=80000)
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和输入规模，不调用 API")
    parser.add_argument("--validate", type=Path, help="只校验已有 JSON 的覆盖率和格式")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有完整或未完成的输出")
    args = parser.parse_args(argv)

    try:
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

        api_url = args.api_url or input(f"API URL [{DEFAULT_API_URL}]: ").strip() or DEFAULT_API_URL
        model = args.model or input(f"模型名 [{DEFAULT_MODEL}]: ").strip() or DEFAULT_MODEL
        endpoint = completion_url(api_url)
        api_key = os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API Key（输入不会回显）：").strip()
        if not api_key:
            raise ValueError("API Key 不能为空")
        if not model.strip():
            raise ValueError("模型名不能为空")
        output = args.output.resolve()
        partial = output.with_name(output.name + ".partial")
        metadata_path = output.with_name(output.name + ".partial.meta.json")
        signature = hashlib.sha256("\n".join(s.sample_id for s in samples).encode()).hexdigest()
        metadata = {"dataset": str(args.dataset.resolve()), "track": args.track,
                    "limit": args.limit, "model": model, "endpoint": endpoint,
                    "sample_ids_sha256": signature}
        if args.overwrite:
            for path in (output, partial, metadata_path):
                path.unlink(missing_ok=True)
        if output.exists():
            raise ValueError(f"完整输出已存在：{output}；更换 --output 或使用 --overwrite")
        completed: list[dict] = []
        if partial.exists() or metadata_path.exists():
            if not (partial.exists() and metadata_path.exists()):
                raise ValueError("续跑文件不完整；请人工检查，或使用 --overwrite")
            if read_json(metadata_path) != metadata:
                raise ValueError("续跑文件来自不同数据/模型/URL；请更换 --output 或使用 --overwrite")
            completed = read_json(partial)
            if not isinstance(completed, list):
                raise ValueError("续跑文件不是 JSON 数组")
            completed_ids = [r.get("sample_id") for r in completed if isinstance(r, dict)]
            if completed_ids != [s.sample_id for s in samples[:len(completed)]]:
                raise ValueError("续跑文件的问题顺序与当前选择不一致")
            for record, sample in zip(completed, samples):
                _, allowed_ids, _ = build_input(sample, args.max_input_chars)
                if record != parse_prediction(json.dumps(record, ensure_ascii=False), sample, allowed_ids):
                    raise ValueError(f"续跑文件中 {sample.sample_id} 格式无效")
            print(f"从 {len(completed)} 条已保存结果继续")
        else:
            atomic_json(metadata_path, metadata)

        for index, sample in enumerate(samples[len(completed):], len(completed) + 1):
            content, allowed_ids, edges = build_input(sample, args.max_input_chars)
            raw = call_model(endpoint, api_key, model, content, args.timeout,
                             args.retries, args.max_tokens)
            record = parse_prediction(raw, sample, allowed_ids)
            chain = record["evidence_chain"]
            if edges and any((a, b) not in edges for a, b in zip(chain, chain[1:])):
                print(f"提示：{sample.sample_id} 的证据链含未在给定图中直接列出的相邻边，请人工复核。", file=sys.stderr)
            completed.append(record)
            atomic_json(partial, completed)
            print(f"[{index}/{len(samples)}] {sample.sample_id} 已保存", flush=True)

        validate_file(partial, samples, args.max_input_chars)
        os.replace(partial, output)
        metadata_path.unlink(missing_ok=True)
        print(f"完成：{output}。请在正式提交前核对赛事对 API 模型的使用规则。")
        return 0
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
