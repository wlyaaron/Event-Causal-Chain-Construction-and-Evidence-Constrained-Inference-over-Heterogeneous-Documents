"""Conservative formatting of an already produced Task 4 refusal.

This module never decides whether the question is answerable or supplies an
evidence ID. Callers can retain the raw response and returned repair tag for
an audit trail.
"""
from __future__ import annotations

import json
import re


PURE_REFUSAL_PREFIXES = (
    "无法回答", "不能确定", "不能得出唯一确定结论",
    "无法精确确定", "不能精确确定", "无法精确量化", "不能精确量化",
    "现有材料不足以确定", "材料不足以确定", "证据不足以确定",
)
EVENT_ID = re.compile(r"\bD\d{3,}\b")


def normalize_raw(raw: str) -> tuple[str, str | None]:
    """Canonicalize clear empty-chain refusals; leave every other output alone.

    The returned tag makes the correction auditable. Invalid JSON and
    non-refusal answers are deliberately not repaired.
    """
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw, None
    if not isinstance(value, dict):
        return raw, None
    answer = value.get("answer")
    chain = value.get("evidence_chain")
    if not isinstance(answer, str) or not isinstance(chain, list):
        return raw, None
    if answer.strip() == "无法确定":
        if chain == [] and value.get("confidence") is None:
            return raw, None
        value.update(answer="无法确定", evidence_chain=[], confidence=None)
        return json.dumps(value, ensure_ascii=False), "exact_refusal_fields"
    # An expanded "无法确定。D003..." may be an attempted factual or
    # counterfactual answer. Keep it for review rather than turning it into a
    # strict refusal simply because it begins with those words.
    if (chain == [] and not EVENT_ID.search(answer)
            and answer.strip().startswith(PURE_REFUSAL_PREFIXES)):
        value.update(answer="无法确定", confidence=None)
        return json.dumps(value, ensure_ascii=False), "empty_chain_refusal_phrase"
    return raw, None
