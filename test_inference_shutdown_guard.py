"""Mock the unattended inference wrapper; never execute the real shutdown."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments import inference_shutdown_guard as guard
from experiments.train_shutdown_guard import GpuSample, OUTPUTS_ROOT


class InferenceShutdownGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        (OUTPUTS_ROOT / "temp").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUTS_ROOT / "temp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.output = self.root / "blind.json"
        self.shutdown = self.root / "shutdown"
        self.shutdown.write_text("test only", encoding="utf-8")

    def run_case(self, return_code: int) -> tuple[int, dict]:
        class FakeProcess:
            pid = 12345

            def __init__(self, *args, **kwargs):
                pass

            def wait(self):
                return return_code

        class ReadyGate:
            def __init__(self, *args):
                pass

            def observe(self, *args):
                return True, 600.0

        argv = ["guard", "--output", str(self.output), "--model-dir", str(self.model),
                "--arm-shutdown"]
        progress = {"signature": {"mode": "blind"},
                    "records": [{}] * 760,
                    "details": [{"format_fallback": False}] * 760}
        with (patch.object(sys, "argv", argv),
              patch.object(guard.os, "geteuid", return_value=0, create=True),
              patch.object(guard, "SHUTDOWN", self.shutdown),
              patch.object(guard.core, "discover_samples", return_value=[
                  SimpleNamespace(track="A") for _ in range(760)]),
              patch.object(guard.core, "validate_file"),
              patch.object(guard.core, "read_json", return_value=progress),
              patch.object(guard.subprocess, "Popen", FakeProcess),
              patch.object(guard, "IdleGate", ReadyGate),
              patch.object(guard, "query_gpu", return_value=GpuSample(0, 0, ())),
              patch.object(guard.subprocess, "run") as shutdown_call):
            outcome = guard.main()
            shutdown_call.assert_called_once_with(
                ["/bin/sh", str(self.shutdown)], check=True, timeout=30)
        status = json.loads((self.root / "blind.guard.json").read_text(encoding="utf-8"))
        return outcome, status

    def test_complete_output_can_request_shutdown(self) -> None:
        outcome, status = self.run_case(0)
        self.assertEqual(outcome, 0)
        self.assertEqual(status["phase"], "shutdown_requested")
        self.assertTrue(status["success"])

    def test_failed_run_records_failure_before_shutdown(self) -> None:
        outcome, status = self.run_case(1)
        self.assertEqual(outcome, 1)
        self.assertEqual(status["phase"], "shutdown_requested_after_failure")
        self.assertFalse(status["success"])


if __name__ == "__main__":
    unittest.main()
