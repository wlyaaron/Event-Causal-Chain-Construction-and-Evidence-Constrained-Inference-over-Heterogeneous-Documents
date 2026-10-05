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


def training_view_input(sample: core.Sample, view: str) -> tuple[str, set[str], set[tuple[str, str]]]:
    """Simulate the three published inference views without reading gold labels."""
    if sample.track != "train" or view not in {"A", "B", "C"}:
        raise ValueError("training_view_input requires a train sample and A/B/C view")
    documents, events, edges = core.load_pack(sample.pack)
    if view == "C":
        events, edges = None, None
    elif view == "B":
        edges = None
    allowed = ({event["event_id"] for event in events} if events is not None
               else set(documents))
    directed = {(edge["cause_event_id"],
                 edge.get("result_event_id", edge.get("effect_event_id")))
                for edge in edges or [] if isinstance(edge, dict)
                and isinstance(edge.get("cause_event_id"), str)
                and isinstance(edge.get("result_event_id", edge.get("effect_event_id")), str)}
    payload = {"track": view, "sample_id": sample.sample_id,
               "question_type": sample.question_type, "question": sample.question,
               "documents": [{"doc_id": doc_id, "text": text}
                             for doc_id, text in documents.items()],
               "events": events, "causal_relations": edges}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), allowed, directed


def relation_guided_input(sample: core.Sample, view: str) -> tuple[str, set[str], dict]:
    """Explain supplied edges and ask for one question-relevant path, not all paths."""
    original, allowed, directed = training_view_input(sample, view)
    if view != "A":
        return original, allowed, {"relation_count": 0, "candidate_paths": []}
    documents, events, edges = core.load_pack(sample.pack)
    labels = {event["event_id"]: str(event.get("event_type") or
                                      event.get("trigger_word") or event["event_id"])
              for event in events or []}
    relations = [f"{a}（{labels.get(a, a)}）导致/推动{b}（{labels.get(b, b)}）"
                 for a, b in sorted(directed)]
    paths = graph_path_hints(sample, limit=5)
    guidance = {
        "given_causal_relations_in_words": relations,
        "question_relevant_candidate_paths": paths,
        "evidence_sufficiency_check": (
            "先找出问题所问的起点、终点及限定条件；逐步核对每条边的方向和原文依据。"
            "候选路径仅供选择，只输出与问题最相关且证据充分的一条。"
            "若问完整过程，保留必要中间环节；若问直接关系，勿扩展无关下游。"
            "给定图未列出某边不必然代表原文无因果，但不能把时间先后当作因果。"
            "遇到冲突或缺口，先核对全部原文，再决定限定回答或拒答。")}
    return ("关系与路径阅读辅助（须以完整原始材料核实）：\n"
            + json.dumps(guidance, ensure_ascii=False)
            + "\n\n完整原始输入：\n" + original), allowed, {
                "relation_count": len(relations), "candidate_paths": paths,
                "input_chars": len(original)}


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


def question_conditioned_paths(sample: core.Sample, limit: int = 5,
                               max_hops: int = 9) -> list[list[str]]:
    """Rank directed local path segments by the question, without reading gold.

    Every prefix is eligible, including paths ending at an internal graph node.
    The graph is a source of candidates, not a guarantee of textual support.
    """
    if limit < 1 or max_hops < 1:
        raise ValueError("limit and max_hops must be positive")
    pack = baseline.read_pack(sample.pack)
    outgoing, _ = baseline.directed_graph(pack)
    if not pack.edges or not outgoing:
        return []
    nodes = set(outgoing)
    anchors = list(dict.fromkeys(node for node in
                                 baseline.ID_PATTERN.findall(sample.question)
                                 if node in nodes))
    question = sample.question
    if anchors and re.search(r"直接(下游|后续|结果)|下一个事件", question):
        intent = "direct_down"
    elif anchors and re.search(r"直接(上游|原因|前因)", question):
        intent = "direct_up"
    elif len(anchors) >= 2 and sample.question_type == "counterfactual":
        intent = "from_to"
    elif anchors and sample.question_type == "prospective":
        intent = "future"
    elif anchors and sample.question_type == "retrospective":
        intent = "upstream"
    else:
        intent = "general"

    ranking = baseline.rank_candidates(question, pack, None, hybrid=False)
    relevance = {candidate.event_id: score for candidate, score in ranking}
    preferred = ([anchors[-1]] if intent == "future" else anchors)
    starts = [*preferred, *sorted(nodes - set(preferred),
                                  key=lambda node: (-relevance.get(node, 0), node))]
    paths: list[list[str]] = []
    max_paths = 20000

    def visit(path: list[str]) -> None:
        if len(paths) >= max_paths:
            return
        if len(path) > 1:
            paths.append(path)
        if len(path) - 1 >= max_hops:
            return
        next_nodes = sorted(outgoing[path[-1]],
                            key=lambda node: (-relevance.get(node, 0), node))
        for next_node in next_nodes:
            if next_node not in path:
                visit([*path, next_node])

    for start in starts:
        visit([start])
        if len(paths) >= max_paths:
            break

    anchor_set = set(anchors)

    def score(path: list[str]) -> tuple:
        if intent == "direct_down":
            match = int(path[0] == anchors[0] and len(path) == 2)
        elif intent == "direct_up":
            match = int(path[-1] == anchors[0] and len(path) == 2)
        elif intent == "from_to":
            match = int(path[0] == anchors[0] and path[-1] == anchors[-1])
        elif intent == "future":
            match = int(path[0] == anchors[-1] and path[-1] not in anchor_set)
        elif intent == "upstream":
            match = int(path[-1] == anchors[0])
        else:
            match = 0
        coverage = len(anchor_set.intersection(path)) / max(1, len(anchor_set))
        relevance_mean = sum(relevance.get(node, 0.0) for node in path) / len(path)
        # Local chains avoid appending unrelated downstream events. Multi-hop
        # questions may still prefer a longer path when its anchors match.
        length_preference = (len(path) if re.search(r"完整|多跳|连锁|传导", question)
                             else -len(path))
        return match, coverage, relevance_mean, length_preference

    return sorted(paths, key=score, reverse=True)[:limit]


def causal_path_input(sample: core.Sample, view: str = "A") -> tuple[str, set[str], dict]:
    """Provide typed, question-ranked causal paths with source excerpts."""
    if sample.track == "train":
        original, allowed, _ = training_view_input(sample, view)
    else:
        if sample.track != view:
            raise ValueError("inference view must match the sample track")
        original, allowed, _ = core.build_input(sample)
    if view != "A":
        return original, allowed, {"candidate_paths": [], "relation_count": 0}
    documents, events, edges = core.load_pack(sample.pack)
    by_id = {event["event_id"]: event for event in events or []}
    by_edge = {(edge.get("cause_event_id"),
                edge.get("result_event_id", edge.get("effect_event_id"))): edge
               for edge in edges or []}
    paths = question_conditioned_paths(sample)
    used_nodes = list(dict.fromkeys(node for path in paths for node in path))
    evidence = []
    for node in used_nodes:
        event = by_id.get(node, {})
        doc_id = event.get("doc_id")
        label = str(event.get("event_type") or event.get("trigger_word") or node)
        evidence.append({"event_id": node, "doc_id": doc_id, "event_label": label,
                         "source_excerpt": _best_excerpt(
                             sample.question, label, documents.get(doc_id, ""),
                             max_chars=160)})
    path_details = []
    for path in paths:
        steps = []
        for cause, result in zip(path, path[1:]):
            edge = by_edge.get((cause, result), {})
            steps.append({"cause": cause, "result": result,
                          "relation_type": edge.get("causal_type",
                                                    edge.get("relation_type")),
                          "confidence_level": edge.get("confidence_level")})
        path_details.append({"event_ids": path, "given_edges": steps})
    guidance = {
        "candidate_status": "题目条件化的图路径候选，未证明原文充分支持，也可能遗漏金标路径",
        "candidate_paths": path_details,
        "event_source_index": evidence,
        "verification": (
            "先区分题目所问的起点、终点、直接关系和已知截止条件；"
            "逐步核对边的方向、关系类型与原文，不把时间先后当因果。"
            "只选一条符合题意且证据充分的链；若候选不合适，可据完整原文与合法 ID 另选。"
            "答案只陈述所选链及原文支持的事实；对错误前提、冲突或缺口先解释再决定是否拒答。")}
    prompt = ("因果路径核验索引（非标准答案）：\n"
              + json.dumps(guidance, ensure_ascii=False, separators=(",", ":"))
              + "\n\n完整原始输入：\n" + original)
    return prompt, allowed, {"candidate_paths": paths,
                             "relation_count": sum(len(p) - 1 for p in paths),
                             "input_chars": len(prompt)}


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
