"""Offline tests. No real API key or paid API call is used."""

import json
import hashlib
import io
import os
import shutil
import threading
import unittest
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import run_deepseek as app
import evaluate_baselines as evaluator


@contextmanager
def temporary_workspace():
    base = Path(__file__).resolve().parent
    path = base / f"test_tmp_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != base:
            raise RuntimeError("test path escaped its workspace")
        shutil.rmtree(path)


def make_pack(root: Path, track: str, questions: int = 1) -> Path:
    pack = root / f"测试集{track}_测试" / "主题_01"
    pack.mkdir(parents=True)
    (pack / "D001.txt").write_text("暴雨导致道路积水。", encoding="utf-8")
    (pack / "D002.txt").write_text("道路积水导致交通中断。", encoding="utf-8")
    (pack / "问题.json").write_text(json.dumps([
        {"sample_id": f"{track}_Q{i:02d}", "question_type": "retrospective",
         "question": "交通中断的原因是什么？"}
        for i in range(1, questions + 1)
    ], ensure_ascii=False), encoding="utf-8")
    if track == "A":
        (pack / "事件列表.json").write_text(json.dumps([
            {"event_id": "D001", "doc_id": "D001", "event_type": "积水"},
            {"event_id": "D002", "doc_id": "D002", "event_type": "中断"},
        ], ensure_ascii=False), encoding="utf-8")
        (pack / "事件因果关系列表.json").write_text(json.dumps([
            {"cause_event_id": "D001", "effect_event_id": "D002"}
        ], ensure_ascii=False), encoding="utf-8")
    gold = pack / "gold"
    gold.mkdir()
    (gold / "问答对_答案.json").write_text("绝不能发给 API 的标准答案", encoding="utf-8")
    return pack


class BaselineTests(unittest.TestCase):
    def test_api_key_file_is_read_without_displaying_secret(self):
        with temporary_workspace() as root:
            key_file = root / "key.txt"
            key_file.write_text("\ufeffsk-test-only-secret\n", encoding="utf-8")
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "wrong-env-key"}):
                with patch.object(app, "probe_balance") as balance:
                    with patch.object(app, "call_model", side_effect=["OK", '{"ok":true}']) as calls:
                        with patch("sys.stdout", new_callable=io.StringIO) as output:
                            result = app.main(["--probe-api", "--api-url", "https://api.deepseek.com",
                                               "--model", "deepseek-flash", "--api-key-file", str(key_file)])
            self.assertEqual(result, 0)
            balance.assert_called_once_with("sk-test-only-secret", 120)
            self.assertEqual(calls.call_args_list[0].args[1], "sk-test-only-secret")
            self.assertNotIn("sk-test-only-secret", output.getvalue())

    def test_local_api_does_not_require_key(self):
        args = app.argparse.Namespace(api_url="http://127.0.0.1:8000/v1", model="local-model",
                                      api_key_file=None)
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(app.getpass, "getpass", side_effect=AssertionError("prompted")):
                endpoint, api_key, model = app.read_api_settings(args)
        self.assertEqual(endpoint, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(api_key, "EMPTY")
        self.assertEqual(model, "local-model")

    def test_empty_http_error_has_useful_diagnostics(self):
        error = HTTPError("https://api.deepseek.com/chat/completions", 400,
                          "Bad Request", {}, io.BytesIO(b""))
        text = app.http_error_text(error, "secret-key")
        self.assertIn("API HTTP 400", text)
        self.assertIn("服务器未返回错误正文", text)
        self.assertNotIn("secret-key", text)

    def test_probe_checks_auth_and_two_minimal_requests_without_dataset(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
            with patch.object(app, "probe_balance") as balance:
                with patch.object(app, "call_model", side_effect=["OK", '{"ok":true}']) as calls:
                    result = app.main(["--probe-api", "--api-url", "https://api.deepseek.com",
                                       "--model", "deepseek-flash", "--dataset", "missing"])
        self.assertEqual(result, 0)
        balance.assert_called_once()
        self.assertEqual(calls.call_count, 2)
        self.assertFalse(calls.call_args_list[0].kwargs["json_mode"])
        self.assertTrue(calls.call_args_list[1].kwargs["json_mode"])

    def test_inputs_follow_actual_track_files_and_exclude_gold(self):
        with temporary_workspace() as root:
            for track in "ABC":
                make_pack(root, track)
            samples = app.discover_samples(root, limit=0)
            self.assertEqual({s.track for s in samples}, {"A", "B", "C"})
            for sample in samples:
                content, allowed, edges = app.build_input(sample)
                self.assertNotIn("绝不能发给 API", content)
                self.assertEqual(allowed, {"D001", "D002"})
                self.assertEqual(bool(edges), sample.track == "A")

    def test_submission_validation_rejects_invalid_evidence(self):
        with temporary_workspace() as root:
            make_pack(root, "A")
            sample = app.discover_samples(root)[0]
            good = app.parse_prediction('{"answer":"暴雨导致积水","evidence_chain":["D001","D002"],"confidence":0.7}', sample, {"D001", "D002"})
            self.assertEqual(good["question_type"], "retrospective")
            with self.assertRaisesRegex(ValueError, "证据 ID"):
                app.parse_prediction('{"answer":"事故","evidence_chain":["X999"],"confidence":0.7}', sample, {"D001", "D002"})
            with self.assertRaisesRegex(ValueError, "拒答时"):
                app.parse_prediction('{"answer":"无法确定","evidence_chain":["D001"],"confidence":null}', sample, {"D001", "D002"})

    def test_unanswerable_submission_uses_null_question_type(self):
        with temporary_workspace() as root:
            pack = make_pack(root, "B")
            (pack / "问题.json").write_text(json.dumps([{
                "sample_id": "B_U01", "question_type": "unanswerable",
                "question": "没有材料的问题？"
            }], ensure_ascii=False), encoding="utf-8")
            sample = app.discover_samples(root)[0]
            record = app.parse_prediction('{"answer":"无法确定","evidence_chain":[],"confidence":null}',
                                          sample, {"D001", "D002"})
            self.assertIsNone(record["question_type"])
            output = root / "submission.json"
            app.atomic_json(output, [record])
            self.assertEqual(app.validate_file(output, [sample], 80000), 1)
            explained = app.parse_prediction('{"answer":"原文证明题目存在错误前提","evidence_chain":["D001"],"confidence":0.5}',
                                             sample, {"D001", "D002"})
            self.assertIsNone(explained["question_type"])
            normalized = app.parse_prediction('{"answer":"无法确定：材料不足","evidence_chain":[],"confidence":0.5}',
                                              sample, {"D001", "D002"})
            self.assertEqual(normalized["answer"], "无法确定")
            self.assertIsNone(normalized["confidence"])

    def test_train_evaluator_reports_answer_and_chain_separately(self):
        with temporary_workspace() as root:
            pack = make_pack(root, "A")
            (pack / "gold" / "问答对_答案.json").write_text(json.dumps([{
                "sample_id": "A_Q01", "answers": "暴雨导致积水，积水导致交通中断。",
                "evidence_chains": [["D001", "D002"]],
                "answer_facts": {"required_facts": ["积水"], "forbidden_facts": []}
            }], ensure_ascii=False), encoding="utf-8")
            sample = app.discover_samples(root)[0]
            record = app.parse_prediction('{"answer":"暴雨导致积水，积水导致交通中断。","evidence_chain":["D001","D002"],"confidence":0.8}',
                                          sample, {"D001", "D002"})
            output = root / "predictions.json"
            app.atomic_json(output, [record])
            result = evaluator.evaluate(output, root)["groups"]["all"]
            self.assertEqual(result["answer_char_f1"], 1.0)
            self.assertEqual(result["chain_edge_f1"], 1.0)
            self.assertEqual(result["required_fact_coverage"], 1.0)

    def test_twice_invalid_model_answer_falls_back_to_valid_refusal(self):
        with temporary_workspace() as root:
            make_pack(root, "A")
            sample = app.discover_samples(root)[0]
            with patch.object(app, "call_model", return_value='{"answer":"猜测","evidence_chain":[],"confidence":0.8}') as model:
                record, unsupported = app.predict(sample, "http://127.0.0.1:1/chat/completions",
                                                  "test-key", "mock", 1, 0, 100, 80000, True)
            self.assertEqual(model.call_count, 2)
            self.assertEqual(record["answer"], "无法确定")
            self.assertFalse(unsupported)

    def test_api_failure_keeps_progress_and_next_run_resumes(self):
        with temporary_workspace() as root:
            make_pack(root, "A", questions=2)
            output = root / "predictions.json"
            state = {"calls": 0, "bodies": []}

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    length = int(self.headers["Content-Length"])
                    body = json.loads(self.rfile.read(length))
                    state["calls"] += 1
                    state["bodies"].append(body)
                    if state["calls"] == 2:
                        self.send_response(503)
                        self.end_headers()
                        self.wfile.write(b"temporary failure")
                        return
                    response = {"choices": [{"finish_reason": "stop", "message": {
                        "content": json.dumps({"answer": "暴雨引起道路积水并导致交通中断",
                                               "evidence_chain": ["D001", "D002"],
                                               "confidence": 0.8}, ensure_ascii=False)
                    }}]}
                    content = json.dumps(response, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)

                def log_message(self, *_):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            args = ["--dataset", str(root), "--limit", "2", "--output", str(output),
                    "--api-url", f"http://127.0.0.1:{server.server_port}",
                    "--model", "mock-model", "--retries", "0"]
            try:
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
                    self.assertEqual(app.main(args), 1)
                    partial = output.with_name(output.name + ".partial")
                    self.assertEqual(len(app.read_json(partial)), 1)
                    self.assertEqual(app.main(args), 0)
                self.assertEqual(state["calls"], 3)
                self.assertEqual(len(app.read_json(output)), 2)
                self.assertEqual(state["bodies"][0]["response_format"], {"type": "json_object"})
                self.assertNotIn("test-only-secret", output.read_text(encoding="utf-8"))
                self.assertEqual(app.validate_file(output, app.discover_samples(root, limit=2), 80000), 2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_old_metadata_without_partial_recovers_after_first_request_failure(self):
        with temporary_workspace() as root:
            make_pack(root, "A")
            output = root / "predictions.json"
            sample = app.discover_samples(root, limit=1)[0]
            metadata = {
                "dataset": str(root.resolve()), "track": "all", "limit": 1,
                "model": "mock-model", "endpoint": "http://127.0.0.1:8123/chat/completions",
                "sample_ids_sha256": hashlib.sha256(sample.sample_id.encode()).hexdigest(),
            }
            app.atomic_json(output.with_name(output.name + ".partial.meta.json"), metadata)
            answer = '{"answer":"暴雨导致交通中断","evidence_chain":["D001","D002"],"confidence":0.8}'
            args = ["--dataset", str(root), "--limit", "1", "--output", str(output),
                    "--api-url", "http://127.0.0.1:8123", "--model", "mock-model"]
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-secret"}):
                with patch.object(app, "call_model", return_value=answer) as call:
                    self.assertEqual(app.main(args), 0)
            call.assert_called_once()
            self.assertEqual(len(app.read_json(output)), 1)


if __name__ == "__main__":
    unittest.main()
