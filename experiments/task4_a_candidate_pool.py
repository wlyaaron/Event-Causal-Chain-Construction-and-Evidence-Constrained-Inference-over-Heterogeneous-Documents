"""Gold-blind A-view causal-chain candidates from the supplied directed graph.

Enumeration follows graph direction, never the numeric order of event IDs.
Ranking is only a relevance heuristic; it does not verify causal truth.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import run_baselines as baseline


# Chinese characters count as word characters in Python, so \b would miss
# strings such as "最终结果D009".
ID_PATTERN = re.compile(r"D\d{3,}")
PROCESS_PATTERN = re.compile(r"一步步|多跳|完整|全程|依次|有序|链条|因果链|传导链")
DIRECT_PATTERN = re.compile(r"直接(?:原因|结果|影响|后续|下游|上游|前因)")
FORWARD_DIRECT_PATTERN = re.compile(
    r"(?:之后.{0,6}直接(?:发生|触发|引发|导致|产生)|"
    r"直接(?:引发|导致|造成|触发|产生|带来|推动)|"
    r"的直接(?:后果|结果))"
)
FACT_SINGLETON_PATTERN = re.compile(
    r"基本事实|关键事实|主体[、，和及].{0,5}地点|记载了哪些"
)
TEMPORAL_CAUSAL_PATTERN = re.compile(
    r"(?:之前|早于|时间先后|从时间看|进程上)"
)
CAUSAL_ASK_PATTERN = re.compile(r"因果|导致")
UNSPECIFIED_DIRECT_PATTERN = re.compile(
    r"与.{0,8}(?:某一|哪一)事件.{0,8}(?:存在|构成)直接因果"
)


def enumerate_graph_paths(
    events: Sequence[dict], edges: Sequence[dict], *, max_paths: int = 100_000
) -> list[tuple[str, ...]]:
    """Return every singleton and simple directed path; never silently truncate."""
    ids = {event["event_id"] for event in events}
    if len(ids) != len(events):
        raise ValueError("Duplicate event IDs")
    adjacency: dict[str, set[str]] = {event_id: set() for event_id in ids}
    for edge in edges:
        cause = edge.get("cause_event_id")
        result = edge.get("result_event_id", edge.get("effect_event_id"))
        if cause in ids and result in ids and cause != result:
            adjacency[cause].add(result)

    paths: set[tuple[str, ...]] = set()

    def visit(path: tuple[str, ...]) -> None:
        if path not in paths:
            paths.add(path)
            if len(paths) > max_paths:
                raise ValueError("Graph path safety cap reached; no partial pool returned")
        for next_id in sorted(adjacency[path[-1]]):
            if next_id not in path:
                visit((*path, next_id))

    for event_id in sorted(ids):
        visit((event_id,))
    return sorted(paths, key=lambda path: (len(path), path))


def _event_text(event: dict, documents: Mapping[str, str]) -> str:
    arguments = event.get("argument") or {}
    if isinstance(arguments, dict):
        argument_text = "；".join(str(value) for value in arguments.values())
    else:
        argument_text = str(arguments)
    doc_id = event.get("doc_id", event["event_id"])
    return " ".join((
        str(event.get("event_type") or ""),
        str(event.get("trigger_word") or ""),
        argument_text[:400],
        documents.get(doc_id, "")[:400],
    ))


def _explicit_roles(question: str, ids: set[str]) -> tuple[str | None, str | None]:
    """Use only explicit role wording; other IDs are weak relevance cues."""
    pair = re.search(r"(?:从|由)\s*(D\d{3,}).{0,80}?(?:到|至|导致)\s*(D\d{3,})", question)
    if pair and pair.group(1) in ids and pair.group(2) in ids:
        return pair.group(1), pair.group(2)
    end = re.search(r"(?:最终结果|最终事件|终点|结果)\s*(D\d{3,})", question)
    if end and end.group(1) in ids:
        return None, end.group(1)
    start = re.search(r"(?:起点|从|截至|止于)\s*(D\d{3,})", question)
    if start and start.group(1) in ids:
        return start.group(1), None
    return None, None


def rank_graph_paths(
    question: str,
    events: Sequence[dict],
    documents: Mapping[str, str],
    paths: Sequence[tuple[str, ...]],
    *,
    limit: int | None = None,
) -> list[tuple[str, ...]]:
    """Rank without gold or hard endpoint filtering; preserve path diversity."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not paths:
        return []
    by_id = {event["event_id"]: event for event in events}
    ids = set(by_id)
    explicit = set(ID_PATTERN.findall(question)) & ids
    start, end = _explicit_roles(question, ids)
    semantic_query = ID_PATTERN.sub(" ", question)
    event_ids = sorted(ids)
    lexical = baseline.bm25_scores(
        semantic_query, [_event_text(by_id[event_id], documents) for event_id in event_ids]
    )
    normalizer = max(lexical, default=0.0) or 1.0
    relevance = {event_id: score / normalizer for event_id, score in zip(event_ids, lexical)}
    wants_process = bool(PROCESS_PATTERN.search(question))
    wants_direct = bool(DIRECT_PATTERN.search(question))

    def score(path: tuple[str, ...]) -> float:
        # Explicit roles affect ranking, not candidate generation.
        role = (1.5 if start == path[0] else 0.0) + (1.5 if end == path[-1] else 0.0)
        anchor = (0.6 * len(explicit.intersection(path)) / len(explicit)
                  if explicit and len(explicit) <= 2 else 0.0)
        semantic = (0.45 * relevance.get(path[0], 0.0)
                    + 0.45 * relevance.get(path[-1], 0.0)
                    + 0.10 * max(relevance.get(node, 0.0) for node in path))
        if wants_direct:
            length = 0.3 if len(path) == 2 else 0.0
        elif wants_process:
            length = 0.08 * min(len(path) - 1, 6)
        else:
            length = -0.03 * (len(path) - 1)
        return role + anchor + semantic + length

    base_scores = {path: score(path) for path in paths}
    node_sets = {path: set(path) for path in paths}
    remaining = sorted(paths, key=lambda path: (-base_scores[path], path))
    selected: list[tuple[str, ...]] = []
    while remaining and (limit is None or len(selected) < limit):
        best = max(
            remaining,
            key=lambda path: (
                base_scores[path] - 0.25 * max(
                    (len(node_sets[path] & node_sets[previous])
                     / len(node_sets[path] | node_sets[previous])
                     for previous in selected),
                    default=0.0,
                ),
                -len(path),
                path,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def targeted_candidate_intent(question: str, question_type: str,
                              event_ids: set[str]) -> tuple[str | None, str | None]:
    """Recognize narrow retrospective asks using only question and event IDs."""
    if question_type != "retrospective":
        return None, None
    mentioned_in_order = list(dict.fromkeys(
        event_id for event_id in ID_PATTERN.findall(question) if event_id in event_ids
    ))
    if (len(mentioned_in_order) == 2
            and TEMPORAL_CAUSAL_PATTERN.search(question)
            and CAUSAL_ASK_PATTERN.search(question)):
        return "temporal_causal_pair", ">".join(mentioned_in_order)
    mentioned = set(mentioned_in_order)
    if len(mentioned) != 1:
        return None, None
    anchor = next(iter(mentioned))
    after_anchor = question[question.find(anchor) + len(anchor):]
    if FORWARD_DIRECT_PATTERN.search(after_anchor):
        return "direct_effect", anchor
    if FACT_SINGLETON_PATTERN.search(question) and not PROCESS_PATTERN.search(question):
        return "event_facts", anchor
    if UNSPECIFIED_DIRECT_PATTERN.search(after_anchor):
        return "unspecified_direct", anchor
    return None, None


def rank_graph_paths_targeted(
    question: str,
    question_type: str,
    events: Sequence[dict],
    edges: Sequence[dict],
    documents: Mapping[str, str],
    paths: Sequence[tuple[str, ...]],
    *,
    limit: int | None = None,
) -> list[tuple[str, ...]]:
    """Put clearly requested direct edges or a factual singleton first.

    All other candidates keep the original rank order. The rule only reads
    inference inputs and never creates a path absent from the visible graph.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    event_ids = {event["event_id"] for event in events}
    intent, anchor = targeted_candidate_intent(question, question_type, event_ids)
    if intent is None or anchor is None:
        return rank_graph_paths(question, events, documents, paths, limit=limit)

    path_set = set(paths)
    if intent == "temporal_causal_pair":
        pair = tuple(anchor.split(">"))
        preferred = [pair] if pair in path_set else []
    elif intent in {"direct_effect", "unspecified_direct"}:
        direct_paths = {
            (cause, result)
            for edge in edges
            if (cause := edge.get("cause_event_id"))
            and (result := edge.get("result_event_id", edge.get("effect_event_id")))
            and (intent == "unspecified_direct" and anchor in (cause, result)
                 or intent == "direct_effect" and cause == anchor)
            and "直接" in str(edge.get("causal_type") or "")
        } & path_set
        preferred = rank_graph_paths(question, events, documents,
                                     sorted(direct_paths)) if direct_paths else []
    else:
        preferred = [(anchor,)] if (anchor,) in path_set else []

    if not preferred:
        return rank_graph_paths(question, events, documents, paths, limit=limit)
    if limit is not None and len(preferred) >= limit:
        return preferred[:limit]

    baseline = rank_graph_paths(question, events, documents, paths,
                                limit=None if limit is None else limit + len(preferred))
    preferred_set = set(preferred)
    ranked = preferred + [path for path in baseline if path not in preferred_set]
    return ranked if limit is None else ranked[:limit]
