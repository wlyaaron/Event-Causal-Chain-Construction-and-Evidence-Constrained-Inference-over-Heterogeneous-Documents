"""Focused checks for model-independent evidence highlighting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import task4_core as core
import task4_workflow as workflow


class WorkflowTests(unittest.TestCase):
    def test_question_anchor_is_kept_and_gold_is_never_in_prompt(self) -> None:
        (core.REPO_ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=core.REPO_ROOT / "outputs") as directory:
            pack = Path(directory)
            for number in range(1, 5):
                (pack / f"D{number:03d}.txt").write_text(
                    f"事件 {number} 发生，相关部门记录处置过程。", encoding="utf-8")
            events = [{"event_id": f"D{number:03d}",
                       "doc_id": f"D{number:03d}",
                       "event_type": f"阶段{number}"}
                      for number in range(1, 5)]
            (pack / "事件列表.json").write_text(
                json.dumps(events, ensure_ascii=False), encoding="utf-8")
            (pack / "事件因果关系列表.json").write_text(
                json.dumps([{"cause_event_id": "D001", "result_event_id": "D002"},
                            {"cause_event_id": "D002", "result_event_id": "D004"}],
                           ensure_ascii=False), encoding="utf-8")
            (pack / "gold").mkdir()
            (pack / "gold" / "问答对_答案.json").write_text(
                "SECRET_GOLD_SENTINEL", encoding="utf-8")
            sample = core.Sample(pack, "A", "demo_Q01", "retrospective",
                                 "D004 的上游原因是什么？")
            prompt, hints = workflow.guided_input(sample, top_k=1,
                                                   hint_mode="graph_paths")
            self.assertIn("D004", [hint.event_id for hint in hints])
            self.assertIn("D001", prompt)
            self.assertIn("D002", prompt)
            self.assertNotIn("SECRET_GOLD_SENTINEL", prompt)
            self.assertIn(["D001", "D002", "D004"],
                          workflow.graph_path_hints(sample))


if __name__ == "__main__":
    unittest.main()
