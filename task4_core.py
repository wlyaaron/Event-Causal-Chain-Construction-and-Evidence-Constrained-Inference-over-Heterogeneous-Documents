"""Shared task-4 data loading, prompt construction, and submission validation."""
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "数据集" / "抽样测试集_100"

SYSTEM_PROMPT = """你是赛题4的证据约束问答系统。只依据用户消息给出的文档、事件和因果关系回答；不得使用外部知识补齐材料缺口。
请输出一个 JSON 对象，且仅包含 answer、evidence_chain、confidence 三个字段。
answer：结合原文而不只依赖事件名称，简洁回答问题并覆盖有材料支持的关键事实；若材料不足、互相冲突或问题前提无法确认，可写“无法确定”或给出有证据的限定解释。输入 question_type 为 unanswerable 时尤其要核查错误前提与冲突证据；不能凭题型直接猜答案。反事实问题只作有边界推断，不把可能性说成确定事实。
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


def load_pack(path: Path) -> tuple[dict[str, str], list[dict] | None, list[dict] | None]:
    """Load inference inputs only; gold answers are intentionally excluded."""
    documents = {file.stem: file.read_text(encoding="utf-8")
                 for file in sorted(path.glob("D[0-9]*.txt"))}
    if not documents:
        raise ValueError(f"文档包缺少 D*.txt：{path}")
    event_file = path / "事件列表.json"
    edge_file = path / "事件因果关系列表.json"
    events = read_json(event_file) if event_file.exists() else None
    edges = read_json(edge_file) if edge_file.exists() else None
    if events is not None and not isinstance(events, list):
        raise ValueError(f"事件列表不是数组：{event_file}")
    if edges is not None and not isinstance(edges, list):
        raise ValueError(f"因果关系列表不是数组：{edge_file}")
    return documents, events, edges


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
    document_map, events, edges = load_pack(sample.pack)
    documents = [{"doc_id": doc_id, "text": text}
                 for doc_id, text in document_map.items()]

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
    refusal_prefixes = ("无法确定", "无法回答", "无法精确", "不能确定", "无法仅凭",
                        "现有材料不足", "资料不足", "证据不足", "无法建立因果关系")
    if not chain and answer.startswith(refusal_prefixes):
        answer = "无法确定"
        confidence = None
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
        "question_type": None if sample.question_type == "unanswerable" else sample.question_type,
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


class PredictionStore:
    """Shared atomic checkpoint, resume, and final submission validation."""

    def __init__(self, output: Path, samples: list[Sample], metadata: dict,
                 max_input_chars: int = 80000, overwrite: bool = False):
        self.output = output.resolve()
        self.partial = self.output.with_name(self.output.name + ".partial")
        self.meta = self.output.with_name(self.output.name + ".partial.meta.json")
        self.samples = samples
        self.max_input_chars = max_input_chars
        if overwrite:
            for path in (self.output, self.partial, self.meta):
                path.unlink(missing_ok=True)
        if self.output.exists():
            raise ValueError(f"完整输出已存在：{self.output}；更换 --output 或使用 --overwrite")
        if self.meta.exists() or self.partial.exists():
            if not self.meta.exists() or read_json(self.meta) != metadata:
                raise ValueError("续跑参数不同或元数据缺失；请更换输出路径或使用 --overwrite")
            self.completed = read_json(self.partial) if self.partial.exists() else []
            if not isinstance(self.completed, list) or len(self.completed) > len(samples):
                raise ValueError("续跑文件不是有效的结果数组")
            for record, sample in zip(self.completed, samples):
                if not isinstance(record, dict) or record.get("sample_id") != sample.sample_id:
                    raise ValueError("续跑文件的问题顺序与当前选择不一致")
                _, allowed_ids, _ = build_input(sample, max_input_chars)
                expected = parse_prediction(json.dumps(record, ensure_ascii=False), sample, allowed_ids)
                if record != expected:
                    raise ValueError(f"续跑文件中 {sample.sample_id} 格式无效")
            if not self.partial.exists():
                atomic_json(self.partial, self.completed)
        else:
            self.completed = []
            atomic_json(self.meta, metadata)
            atomic_json(self.partial, self.completed)

    def append(self, record: dict) -> None:
        index = len(self.completed)
        if index >= len(self.samples) or record.get("sample_id") != self.samples[index].sample_id:
            raise ValueError("结果顺序与所选问题不一致")
        self.completed.append(record)
        atomic_json(self.partial, self.completed)

    def finish(self) -> int:
        count = validate_file(self.partial, self.samples, self.max_input_chars)
        os.replace(self.partial, self.output)
        self.meta.unlink(missing_ok=True)
        return count
