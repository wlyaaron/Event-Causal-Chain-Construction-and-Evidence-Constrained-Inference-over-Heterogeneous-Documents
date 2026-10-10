"""Gold-blind route parsing and evidence-chain proposals for task 4.

Candidates are hints for source checking, never an automatic final answer.
Only the graph included in the current inference view may be used.
"""
from __future__ import annotations

import json
import re
from itertools import combinations

ROUTES = {"strict_refusal", "limited_explanation", "answer"}
EVIDENCE_MODES = {"none", "anchor", "path"}
ID_PATTERN = re.compile(r"\bD\d{3,}\b")


def parse_route(raw: str, allowed: set[str]) -> dict:
    """Validate the model's intermediate decision without judging its truth."""
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("route") not in ROUTES:
        raise ValueError("Invalid answer route")
    mode = value.get("evidence_mode")
    if mode not in EVIDENCE_MODES:
        raise ValueError("Invalid evidence mode")
    ids = value.get("basis_ids")
    if not isinstance(ids, list) or any(not isinstance(x, str) or x not in allowed for x in ids):
        raise ValueError("Invalid basis IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated basis ID")
    if value["route"] == "strict_refusal" and mode != "none":
        raise ValueError("Strict refusal must use none evidence mode")
    if value["route"] != "strict_refusal" and mode == "none":
        raise ValueError("Non-refusal requires an evidence mode")
    target = value.get("requested_target")
    reason = value.get("reason")
    if not isinstance(target, str) or not target.strip() or not isinstance(reason, str):
        raise ValueError("Route lacks requested target or reason")
    return {"route": value["route"], "evidence_mode": mode,
            "basis_ids": ids, "requested_target": target.strip(), "reason": reason.strip()}


def candidates(payload: dict, draft: dict | None, route: dict, limit: int = 16) -> list[dict]:
    """Keep the draft plus small alternatives; return no paths for strict refusal."""
    if route["route"] == "strict_refusal":
        return []
    events = payload.get("events")
    allowed = ({e["event_id"] for e in events} if isinstance(events, list)
               else {d["doc_id"] for d in payload["documents"]})
    selected: list[dict] = []
    seen: set[tuple[str, ...]] = set()

    def add(chain: list[str], source: str) -> None:
        key = tuple(chain)
        if key and key not in seen and len(key) == len(set(key)) and set(key) <= allowed:
            seen.add(key)
            selected.append({"chain": chain, "source": source})

    original = (draft or {}).get("evidence_chain") or []
    if isinstance(original, list) and all(isinstance(x, str) for x in original):
        add(original, "original")
        for left in range(len(original)):
            for right in range(len(original), left, -1):
                if left or right < len(original):
                    add(original[left:right], "contiguous_trim")
        if 3 <= len(original) <= 7:
            for size in range(len(original) - 1, 1, -1):
                for positions in combinations(range(len(original)), size):
                    if positions[-1] - positions[0] + 1 > size:
                        add([original[i] for i in positions], "ordered_skip")

    for node in route["basis_ids"]:
        add([node], "route_anchor")
    for node in ID_PATTERN.findall(payload["question"]):
        add([node], "question_anchor")

    # The A view alone contains causal_relations. B/C never see a hidden graph.
    edges = payload.get("causal_relations")
    if route["evidence_mode"] == "path" and isinstance(edges, list):
        anchors = set(route["basis_ids"]) | set(ID_PATTERN.findall(payload["question"]))
        for edge in edges:
            start = edge.get("cause_event_id")
            end = edge.get("result_event_id", edge.get("effect_event_id"))
            if start in anchors or end in anchors:
                add([start, end], "visible_directed_edge")

    return selected[:limit]
