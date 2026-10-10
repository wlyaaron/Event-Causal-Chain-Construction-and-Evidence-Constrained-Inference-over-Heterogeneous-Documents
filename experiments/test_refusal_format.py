"""Focused tests for refusal formatting without semantic reclassification."""
import json
import unittest
from pathlib import Path

import task4_core as core
from task4_refusal_format import normalize_raw


class RefusalFormatTests(unittest.TestCase):
    def normalize(self, value: dict) -> tuple[dict, str | None]:
        raw, tag = normalize_raw(json.dumps(value, ensure_ascii=False))
        return json.loads(raw), tag

    def test_empty_chain_variant_becomes_exact_refusal(self) -> None:
        value, tag = self.normalize({"answer": "不能得出唯一确定结论。口径尚未统一。",
                                     "evidence_chain": [], "confidence": 0.7})
        self.assertEqual(value, {"answer": "无法确定", "evidence_chain": [],
                                 "confidence": None})
        self.assertEqual(tag, "empty_chain_refusal_phrase")

    def test_exact_refusal_clears_inconsistent_fields(self) -> None:
        value, tag = self.normalize({"answer": "无法确定", "evidence_chain": ["D001"],
                                     "confidence": 0.8})
        self.assertEqual(value["evidence_chain"], [])
        self.assertIsNone(value["confidence"])
        self.assertEqual(tag, "exact_refusal_fields")

    def test_evidence_bearing_limited_explanation_is_preserved(self) -> None:
        original = {"answer": "无法确定精确数值，但 D003 给出冲突口径。",
                    "evidence_chain": ["D003"], "confidence": 0.6}
        value, tag = self.normalize(original)
        self.assertEqual(value, original)
        self.assertIsNone(tag)

    def test_answer_without_chain_is_not_invented(self) -> None:
        original = {"answer": "D002 是直接后果。", "evidence_chain": [],
                    "confidence": 0.8}
        value, tag = self.normalize(original)
        self.assertEqual(value, original)
        self.assertIsNone(tag)

    def test_refusal_with_event_claim_is_not_silently_reclassified(self) -> None:
        original = {"answer": "无法确定。移除 D003 后 D006 仍可能发生。",
                    "evidence_chain": [], "confidence": None}
        value, tag = self.normalize(original)
        self.assertEqual(value, original)
        self.assertIsNone(tag)

    def test_invalid_json_is_preserved(self) -> None:
        self.assertEqual(normalize_raw('{"answer":'), ('{"answer":', None))

    def test_core_parser_uses_conservative_format_rule(self) -> None:
        sample = core.Sample(Path("unused"), "A", "sample", "unanswerable", "question")
        fixed = core.parse_prediction(json.dumps({
            "answer": "不能得出唯一确定结论。口径尚未统一。",
            "evidence_chain": [], "confidence": 0.7,
        }, ensure_ascii=False), sample, {"D003", "D006"})
        self.assertEqual((fixed["answer"], fixed["evidence_chain"], fixed["confidence"]),
                         ("无法确定", [], None))
        with self.assertRaisesRegex(ValueError, "可回答题必须提供证据链"):
            core.parse_prediction(json.dumps({
                "answer": "无法确定。移除 D003 后 D006 仍可能发生。",
                "evidence_chain": [], "confidence": None,
            }, ensure_ascii=False), sample, {"D003", "D006"})


if __name__ == "__main__":
    unittest.main()
