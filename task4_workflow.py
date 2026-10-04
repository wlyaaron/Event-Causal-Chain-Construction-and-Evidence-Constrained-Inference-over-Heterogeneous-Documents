"""Model-independent evidence highlighting for task-4 generation.

The highlighted candidates are retrieval hints, never a replacement for the
original supplied documents. No gold answer is read here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import run_baselines as baseline
import task4_core as core


@dataclass(frozen=True)
class EvidenceHint:
    event_id: str
    doc_id: str
    label: str
    excerpt: str
    source: str


def _best_excerpt(question: str, label: str, document: str,
                  max_chars: int = 240) -> str:
    parts = [item.strip() for item in re.split(r"(?<=[。！？；;])|\n+", document)
             if item.strip()]
    if not parts:
        return ""
    scores = baseline.bm25_scores(question + " " + label, parts)
    best = max(range(len(parts)), key=lambda index: scores[index])
    return parts[best][:max_chars]


def retrieve_hints(sample: core.Sample, top_k: int = 3,
                   encoder: baseline.BertEncoder | None = None,
                   ranked_ids: list[str] | None = None) -> list[EvidenceHint]:
    """Add valid question-mentioned IDs to ranked candidates without a hard cap."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    pack = baseline.read_pack(sample.pack)
    by_id = {candidate.event_id: candidate for candidate in pack.candidates}
    if ranked_ids is None:
        ranking = baseline.rank_candidates(sample.question, pack, encoder,
                                           hybrid=encoder is not None)
        ranked_ids = [candidate.event_id for candidate, _ in ranking]
    elif len(ranked_ids) != len(by_id) or set(ranked_ids) != set(by_id):
        raise ValueError("ranked_ids must contain every candidate exactly once")
    anchors = [event_id for event_id in baseline.ID_PATTERN.findall(sample.question)
               if event_id in by_id]
    chosen = []
    seen = set()
    for event_id in [*ranked_ids[:top_k], *anchors]:
        if event_id in seen:
            continue
        seen.add(event_id)
        candidate = by_id[event_id]
        chosen.append(EvidenceHint(
            event_id=event_id,
            doc_id=candidate.doc_id,
            label=candidate.label,
            excerpt=_best_excerpt(sample.question, candidate.label,
                                  pack.documents[candidate.doc_id]),
            source="ranked+question_anchor" if event_id in anchors else "ranked",
        ))
    return chosen


def graph_path_hints(sample: core.Sample, limit: int = 3) -> list[list[str]]:
    """Propose paths from supplied graph topology; never certify them as gold."""
    pack = baseline.read_pack(sample.pack)
    if not pack.edges:
        return []
    outgoing, incoming = baseline.directed_graph(pack)
    if not outgoing:
        return []
    anchors = set(baseline.ID_PATTERN.findall(sample.question)) & set(outgoing)
    ranks = baseline.rank_candidates(sample.question, pack, None, hybrid=False)
    relevance = {candidate.event_id: score for candidate, score in ranks}
    roots = [node for node in outgoing if not incoming[node]] or list(outgoing)
    paths = []

    def visit(node: str, path: list[str]) -> None:
        if len(paths) >= 5000:
            return
        if not outgoing[node] or len(path) >= 12:
            paths.append(path)
            return
        for next_node in outgoing[node]:
            if next_node not in path:
                visit(next_node, [*path, next_node])

    for root in roots:
        visit(root, [root])
    scored = sorted(paths, key=lambda path: (
        len(anchors.intersection(path)) / max(1, len(anchors)),
        sum(relevance.get(node, 0.0) for node in path) / len(path),
        len(path),
    ), reverse=True)
    selected = []
    for path in scored:
        if path not in selected:
            selected.append(path)
        if len(selected) >= limit:
            break
    return selected


def guided_input(sample: core.Sample, top_k: int = 3,
                 encoder: baseline.BertEncoder | None = None,
                 hint_mode: str = "nodes",
                 ranked_ids: list[str] | None = None) -> tuple[str, list[EvidenceHint]]:
    """Give a compact evidence index first, then all authoritative source data."""
    if hint_mode not in {"direct", "nodes", "graph_paths", "learned"}:
        raise ValueError(f"Unsupported hint mode: {hint_mode}")
    original, _, _ = core.build_input(sample)
    if hint_mode == "direct":
        return original, []
    hints = retrieve_hints(sample, top_k, encoder, ranked_ids)
    index = {
        "retrieval_hint_status": "候选线索，可能漏掉关键事件；原始材料才是依据",
        "candidate_evidence": [hint.__dict__ for hint in hints],
        "instructions": (
            "先依据原始材料核实问题涉及的原因、结果和中间事件，再从原因到结果选择一条有据可查的链。"
            "若问题要求完整传导或预测路径，应从题目所问的起点到目标保留必要的上游与中间环节，"
            "不要只列结果附近的末端节点；若只问直接关系，则不要添加无关下游事件。"
            "若候选遗漏关键证据，可使用原始材料中的其他合法 ID；时间先后不等于因果。"
            "答案要覆盖关键事实，避免材料外补充；无充分证据才拒答。最终只输出指定 JSON 对象。"
        ),
    }
    if hint_mode == "graph_paths":
        index["candidate_paths"] = graph_path_hints(sample)
        index["path_note"] = ("这些路径由给定边搜索得出，只是候选；"
                              "须结合问题和原文选择，也允许原文支持的其他合法路径。")
    return ("证据检索索引：\n" + json.dumps(index, ensure_ascii=False)
            + "\n\n完整原始输入：\n" + original), hints
