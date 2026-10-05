"""Focused checks for model-independent evidence highlighting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import task4_core as core
import task4_workflow as workflow
from experiments.run_full_causal_path_research import input_for


class WorkflowTests(unittest.TestCase):
    def test_full_research_changes_only_a_view(self) -> None:
        for track in "ABC":
            sample = core.discover_samples(core.DEFAULT_DATASET, track, 1)[0]
            original, allowed, _ = core.build_input(sample)
            content, selected_allowed = input_for(sample)
            self.assertEqual(allowed, selected_allowed)
            if track == "A":
                self.assertIn("因果路径核验索引", content)
                self.assertIn(original, content)
            else:
                self.assertEqual(original, content)

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

    def test_direct_downstream_uses_internal_segment_and_preserves_type(self) -> None:
        (core.REPO_ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=core.REPO_ROOT / "outputs") as directory:
            pack = Path(directory)
            for number in range(1, 5):
                (pack / f"D{number:03d}.txt").write_text(
                    f"阶段{number}形成下一阶段。", encoding="utf-8")
            events = [{"event_id": f"D{number:03d}",
                       "doc_id": f"D{number:03d}",
                       "event_type": f"阶段{number}"}
                      for number in range(1, 5)]
            edges = [{"cause_event_id": f"D{number:03d}",
                      "result_event_id": f"D{number + 1:03d}",
                      "causal_type": "间接传导" if number == 2 else "直接因果",
                      "confidence_level": "probable"}
                     for number in range(1, 4)]
            (pack / "事件列表.json").write_text(
                json.dumps(events, ensure_ascii=False), encoding="utf-8")
            (pack / "事件因果关系列表.json").write_text(
                json.dumps(edges, ensure_ascii=False), encoding="utf-8")
            (pack / "gold").mkdir()
            (pack / "gold" / "问答对_答案.json").write_text(
                "SECRET_GOLD_SENTINEL", encoding="utf-8")
            sample = core.Sample(pack, "train", "demo_Q02", "retrospective",
                                 "D002 的直接下游事件是什么？")
            paths = workflow.question_conditioned_paths(sample)
            self.assertEqual(["D002", "D003"], paths[0])
            prompt, allowed, meta = workflow.causal_path_input(sample)
            self.assertIn("间接传导", prompt)
            self.assertIn("probable", prompt)
            self.assertNotIn("SECRET_GOLD_SENTINEL", prompt)
            self.assertEqual({f"D{number:03d}" for number in range(1, 5)}, allowed)
            self.assertEqual(["D002", "D003"], meta["candidate_paths"][0])
            test_sample = core.Sample(pack, "A", "demo_Q03", "retrospective",
                                      "D002 的直接下游事件是什么？")
            test_prompt, test_allowed, test_meta = workflow.causal_path_input(test_sample)
            self.assertEqual(allowed, test_allowed)
            self.assertEqual(paths[0], test_meta["candidate_paths"][0])
            self.assertNotIn("SECRET_GOLD_SENTINEL", test_prompt)


if __name__ == "__main__":
    unittest.main()
