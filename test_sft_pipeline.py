"""CPU checks for the SFT data and loss boundary."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


class SftDataTests(unittest.TestCase):
    def test_each_epoch_uses_one_intact_reference(self):
        from experiments.build_sft_mix import mixed_rows

        def row(view):
            return {"pack": "p1", "sample_id": "q1", "view": view,
                    "messages": [{"role": "system", "content": "s"},
                                 {"role": "user", "content": view},
                                 {"role": "assistant", "content": json.dumps({
                                     "answer": "a", "evidence_chain": ["D001", "D002"]})}],
                    "compatible_gold_chains": [["D001", "D002"],
                                               ["D001", "D003"]],
                    "visible_graph_supported_chains": 1 if view == "A" else None,
                    "flags": []}

        source = {view: [row(view)] for view in "ABC"}
        result = list(mixed_rows(source, epochs=3, seed=7, multireference=True,
                                 weights={"A": .6, "B": .3, "C": .1},
                                 exclude_flags=set(), a_graph_supported_only=False))
        self.assertEqual(len(result), 3)
        for item in result:
            target = json.loads(item["messages"][-1]["content"])
            self.assertIn(target["evidence_chain"], item["compatible_gold_chains"])
            self.assertEqual(target["answer"], "a")

    def test_loss_masks_all_prompt_tokens_and_keeps_target(self):
        from experiments.train_sft import encode_supervised

        class TinyTokenizer:
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=False):
                rendered = "|".join(message["content"] for message in messages)
                if add_generation_prompt:
                    rendered += "|"
                return rendered

            def __call__(self, rendered, add_special_tokens=False,
                         return_offsets_mapping=True):
                return {"input_ids": [ord(c) for c in rendered],
                        "offset_mapping": [(i, i + 1) for i in range(len(rendered))]}

        row = {"messages": [{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"}]}
        encoded = encode_supervised(row, TinyTokenizer(), max_tokens=10)
        self.assertEqual(encoded["labels"], [-100, -100, -100, -100, ord("a")])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            encode_supervised(row, TinyTokenizer(), max_tokens=4)

    def test_length_cache_reuses_exact_message_and_keeps_limit_decision(self):
        from experiments.train_sft import prepare_records

        class TinyTokenizer:
            calls = 0

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=False):
                content = "|".join(message["content"] for message in messages)
                if add_generation_prompt:
                    content += "|"
                return content

            def __call__(self, content, add_special_tokens=False,
                         return_offsets_mapping=True):
                self.calls += 1
                return {"input_ids": [ord(char) for char in content],
                        "offset_mapping": [(i, i + 1) for i in range(len(content))]}

        row = {"pack": "p", "sample_id": "q", "view": "A",
               "messages": [{"role": "system", "content": "s"},
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"}]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rows.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with sqlite3.connect(":memory:") as cache:
                cache.execute("CREATE TABLE lengths "
                              "(key TEXT PRIMARY KEY, full_tokens INTEGER, target_tokens INTEGER)")
                tokenizer = TinyTokenizer()
                first = prepare_records(source, tokenizer, 10, False, cache, "fingerprint")
                second = prepare_records(source, tokenizer, 10, False, cache, "fingerprint")
                self.assertEqual(first[:4], second[:4])
                self.assertEqual((first[4], second[4], tokenizer.calls), (0, 1, 1))
                with self.assertRaisesRegex(ValueError, "exceeds 4"):
                    prepare_records(source, tokenizer, 4, False, cache, "fingerprint")

    def test_evaluation_accepts_best_complete_gold_alternative(self):
        from experiments.evaluate_sft import score_row

        gold = {"answers": "事故推动救援", "evidence_chains": [
            ["D001", "D002"], ["D001", "D003", "D004"]],
            "answer_facts": {"required_facts": ["事故"]}}
        prediction = {"answer": "事故推动救援", "evidence_chain":
                      ["D001", "D003", "D004"]}
        scored = score_row(prediction, gold)
        self.assertEqual(scored["chain_exact"], 1.0)
        self.assertEqual(scored["chain_event_f1"], 1.0)
        self.assertEqual(scored["chain_edge_f1"], 1.0)
        self.assertEqual(scored["required_fact_coverage"], 1.0)

    def test_inference_rejects_unknown_evidence_and_false_refusal(self):
        from experiments.run_sft import validate_answer_chain

        self.assertEqual(validate_answer_chain(
            '{"answer":"事故推动救援","evidence_chain":["D001"]}',
            {"D001"})["evidence_chain"], ["D001"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_answer_chain('{"answer":"a","evidence_chain":["D999"]}',
                                  {"D001"})
        with self.assertRaisesRegex(ValueError, "refusal"):
            validate_answer_chain(
                '{"answer":"无法确定","evidence_chain":["D001"]}', {"D001"})
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_answer_chain(
                '{"answer":"a","evidence_chain":["D001"],"confidence":0.9}',
                {"D001"})

    def test_calibration_uses_validation_success_only(self):
        from experiments.calibrate_sft import fit_rates, calibrated_confidence

        records = [{"view": "A", "evidence_chain": ["D001"],
                    "answer": "有依据", "success": True},
                   {"view": "A", "evidence_chain": ["D002"],
                    "answer": "有依据", "success": False}]
        rates = fit_rates(records)
        value = calibrated_confidence(records[0], rates)
        self.assertGreater(value, 0)
        self.assertLess(value, 1)
        self.assertIsNone(calibrated_confidence(
            {"view": "A", "answer": "无法确定", "evidence_chain": []}, rates))


if __name__ == "__main__":
    unittest.main()
