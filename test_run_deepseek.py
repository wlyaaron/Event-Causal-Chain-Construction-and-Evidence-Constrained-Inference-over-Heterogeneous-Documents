"""Offline tests. No real API key or paid API call is used."""

import json
import os
import shutil
import threading
import unittest
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import run_deepseek as app


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


if __name__ == "__main__":
    unittest.main()
