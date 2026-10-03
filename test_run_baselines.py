"""Offline checks for the independently specified competition baselines."""

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import run_baselines as baseline
import run_deepseek as common


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f"baseline_test_{uuid.uuid4().hex}"
        self.pack = self.root / "测试集A_基础" / "主题_01"
        self.pack.mkdir(parents=True)
        (self.pack / "D001.txt").write_text("暴雨导致积水。", encoding="utf-8")
        (self.pack / "D002.txt").write_text("积水导致停运。", encoding="utf-8")
        (self.pack / "D003.txt").write_text("停运后开展整治。", encoding="utf-8")
        (self.pack / "问题.json").write_text(json.dumps([{
            "sample_id": "x_Q01", "question_type": "retrospective",
            "question": "请从起点还原 D003 的完整因果链。"
        }], ensure_ascii=False), encoding="utf-8")
        (self.pack / "事件列表.json").write_text(json.dumps([
            {"event_id": f"D00{i}", "doc_id": f"D00{i}", "event_type": name}
            for i, name in [(1, "暴雨"), (2, "积水"), (3, "停运")]
        ], ensure_ascii=False), encoding="utf-8")
        (self.pack / "事件因果关系列表.json").write_text(json.dumps([
            {"cause_event_id": "D001", "result_event_id": "D002", "confidence_level": "certain"},
            {"cause_event_id": "D002", "result_event_id": "D003", "confidence_level": "certain"},
        ]), encoding="utf-8")
        gold = self.pack / "gold"
        gold.mkdir()
        (gold / "问答对_答案.json").write_text("PRIVATE_GOLD_SENTINEL", encoding="utf-8")

    def tearDown(self):
        if self.root.resolve().parent != Path(__file__).resolve().parent:
            raise RuntimeError("Unexpected test cleanup target")
        shutil.rmtree(self.root)

    def test_bm25_and_graph_emit_ordered_existing_ids(self):
        scores = baseline.bm25_scores("暴雨", ["暴雨导致积水", "停运后整改"])
        self.assertGreater(scores[0], scores[1])
        output = self.root / "graph.json"
        self.assertEqual(baseline.run(["--method", "graph", "--dataset", str(self.root),
                                       "--limit", "1", "--output", str(output)]), 0)
        record = common.read_json(output)[0]
        self.assertEqual(record["evidence_chain"], ["D001", "D002", "D003"])
        self.assertEqual(set(record), {"sample_id", "answer", "evidence_chain",
                                       "confidence", "question_type"})

    def test_rag_retrieval_excludes_gold_and_keeps_question_anchor(self):
        sample = common.discover_samples(self.root)[0]
        pack = baseline.read_pack(self.pack)
        ranking = baseline.rank_candidates(sample.question, pack, None, hybrid=False)
        content = baseline.rag_input(sample, pack, ranking, top_k=1)
        self.assertNotIn("PRIVATE_GOLD_SENTINEL", content)
        payload = json.loads(content)
        self.assertIn("D003", [d["doc_id"] for d in payload["retrieved_documents"]])

    def test_rag_and_llama3_use_same_submission_validator(self):
        response = json.dumps({"answer": "暴雨引发积水并造成停运",
                               "evidence_chain": ["D001", "D002", "D003"],
                               "confidence": 0.8}, ensure_ascii=False)
        for method in ("rag", "llama3"):
            output = self.root / f"{method}.json"
            with patch.object(common, "call_model", return_value=response) as model:
                self.assertEqual(baseline.run(["--method", method,
                                               "--dataset", str(self.root), "--limit", "1",
                                               "--output", str(output), "--api-url",
                                               "http://127.0.0.1:8000/v1", "--model", "local-model"]), 0)
            self.assertEqual(model.call_count, 1)
            self.assertEqual(common.validate_file(output, common.discover_samples(self.root), 80000), 1)

    def test_rag_retries_missing_evidence_chain_without_inventing_it(self):
        invalid = json.dumps({"answer": "暴雨造成停运", "evidence_chain": [],
                              "confidence": 0.8}, ensure_ascii=False)
        valid = json.dumps({"answer": "无法确定", "evidence_chain": [],
                            "confidence": None}, ensure_ascii=False)
        output = self.root / "rag_retry.json"
        with patch.object(common, "call_model", side_effect=[invalid, valid]) as model:
            self.assertEqual(baseline.run(["--method", "rag", "--dataset", str(self.root),
                                           "--limit", "1", "--output", str(output),
                                           "--api-url", "http://127.0.0.1:8000/v1",
                                           "--model", "local-model"]), 0)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(common.read_json(output)[0]["answer"], "无法确定")


if __name__ == "__main__":
    unittest.main()
