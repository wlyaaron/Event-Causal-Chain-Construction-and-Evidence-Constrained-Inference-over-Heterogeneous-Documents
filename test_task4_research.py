"""Checks for view isolation, grouped splits, and intact gold alternatives."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import task4_core as core
import task4_workflow as workflow
from experiments.prepare_finetune_data import (SFT_SYSTEM_PROMPT, choose_chain,
                                                compatible_chains,
                                                document_groups, make_split)
from experiments.sample_multireference_epoch import selected_chain


class ResearchDataTests(unittest.TestCase):
    def test_three_views_hide_only_the_intended_fields(self):
        sample = core.Sample(core.REPO_ROOT / "数据集" / "训练集" / "公共安全_001",
                             "train", "example", "retrospective", "D001到D002为何发生？")
        a, a_ids, a_edges = workflow.training_view_input(sample, "A")
        b, b_ids, b_edges = workflow.training_view_input(sample, "B")
        c, c_ids, c_edges = workflow.training_view_input(sample, "C")
        pa, pb, pc = (json.loads(text) for text in (a, b, c))
        self.assertTrue(pa["events"] and pa["causal_relations"])
        self.assertTrue(pb["events"])
        self.assertIsNone(pb["causal_relations"])
        self.assertIsNone(pc["events"])
        self.assertIsNone(pc["causal_relations"])
        self.assertEqual(a_ids, b_ids)
        self.assertEqual(c_ids, {document["doc_id"] for document in pc["documents"]})
        self.assertTrue(a_edges)
        self.assertFalse(b_edges or c_edges)
        self.assertEqual(pa["question"], pb["question"])
        self.assertEqual(pa["documents"], pc["documents"])
        guided, _, meta = workflow.relation_guided_input(sample, "A")
        self.assertIn("given_causal_relations_in_words", guided)
        self.assertTrue(meta["relation_count"])
        self.assertNotIn("required_facts", guided)

    def test_selects_one_gold_chain_and_preserves_explanatory_unanswerable(self):
        item = {"answers": "前提不成立，D001与D002构成冲突",
                "evidence_chains": [["D001", "D002"], ["D001", "D003"]]}
        chosen = choose_chain(item, {"D001", "D002", "D003"}, "D001与D002是否因果")
        self.assertIn(chosen, item["evidence_chains"])
        self.assertNotEqual(chosen, ["D001", "D002", "D003"])
        self.assertEqual(choose_chain(item, {"D001", "D002", "D003"},
                                      "D001与D002是否因果", {("D001", "D003")}),
                         ["D001", "D003"])
        self.assertEqual(choose_chain({"answers": "无法确定", "evidence_chains": []},
                                      {"D001"}, "?"), [])
        self.assertNotIn('"confidence":', SFT_SYSTEM_PROMPT)

    def test_absent_visible_edge_does_not_erase_gold_or_make_false_negative(self):
        item = {"answers": "D001导致D003", "evidence_chains": [
            ["D001", "D002", "D003"], ["D001", "D003"]]}
        allowed = {"D001", "D002", "D003"}
        self.assertEqual(2, len(compatible_chains(item, allowed)))
        self.assertIn(choose_chain(item, allowed, "D001如何到达D003？",
                                   {("D001", "D002")}), item["evidence_chains"])
        row = {"pack": "测试_001", "sample_id": "Q01", "view": "A",
               "compatible_gold_chains": compatible_chains(item, allowed)}
        self.assertEqual(selected_chain(row, 20261005, 2), selected_chain(row, 20261005, 2))
        self.assertIn(selected_chain(row, 20261005, 2), item["evidence_chains"])

    def test_duplicate_documents_are_grouped_before_split(self):
        with tempfile.TemporaryDirectory(dir=core.REPO_ROOT) as temp:
            root = Path(temp)
            for name, text in (("甲_01", "相同 内容"), ("乙_01", "相同\n内容"),
                               ("乙_02", "另一个文档")):
                pack = root / name
                pack.mkdir()
                (pack / "D001.txt").write_text(text, encoding="utf-8")
            groups = document_groups(sorted(root.iterdir()))
            self.assertIn(["乙_01", "甲_01"], groups)
            split = make_split(groups, 20261005)
            assignment = {pack: name for name, packs in split.items() for pack in packs}
            self.assertEqual(assignment["甲_01"], assignment["乙_01"])


if __name__ == "__main__":
    unittest.main()
