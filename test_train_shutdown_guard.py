"""Safety checks for the unattended training wrapper; never invoke shutdown."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import train_shutdown_guard as guard


class ShutdownGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        (guard.OUTPUTS_ROOT / "temp").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=guard.OUTPUTS_ROOT / "temp")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def make_artifacts(self, step: int = 1443, maximum: int = 1443,
                       output: Path | None = None) -> None:
        destination = output or self.output
        (destination / "adapter_config.json").write_text("{}", encoding="utf-8")
        (destination / "adapter_model.safetensors").write_bytes(b"x" * 2048)
        (destination / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "max_steps": maximum}), encoding="utf-8"
        )
        (destination / "train_results.json").write_text(
            json.dumps({"train_runtime": 12.5, "train_loss": 1.2}), encoding="utf-8"
        )

    def test_complete_run_with_adapter_is_accepted(self) -> None:
        self.make_artifacts()
        self.assertEqual(guard.validate_artifacts(self.output, min_steps=1000)["global_step"], 1443)

    def test_short_or_incomplete_run_cannot_trigger_armed_shutdown(self) -> None:
        self.make_artifacts(step=2, maximum=2)
        with self.assertRaises(ValueError):
            guard.validate_artifacts(self.output, min_steps=1000)
        self.make_artifacts(step=1442, maximum=1443)
        with self.assertRaises(ValueError):
            guard.validate_artifacts(self.output, min_steps=1000)
        (self.output / "adapter_model.safetensors").unlink()
        with self.assertRaises(ValueError):
            guard.validate_artifacts(self.output)

    def test_idle_timer_resets_on_load_memory_or_process(self) -> None:
        gate = guard.IdleGate(utilization_limit=5, memory_limit_mib=2048, idle_seconds=600)
        idle = guard.GpuSample(0, 3, ())
        self.assertFalse(gate.observe(0, idle)[0])
        self.assertFalse(gate.observe(590, idle)[0])
        self.assertFalse(gate.observe(600, guard.GpuSample(6, 3, ()))[0])
        self.assertFalse(gate.observe(601, idle)[0])
        self.assertFalse(gate.observe(1200, guard.GpuSample(0, 3000, ()))[0])
        self.assertFalse(gate.observe(1201, guard.GpuSample(0, 3, ("1234",)))[0])
        self.assertFalse(gate.observe(1202, idle)[0])
        self.assertTrue(gate.observe(1802, idle)[0])

    def test_nvidia_smi_parse_requires_no_compute_processes(self) -> None:
        class Result:
            def __init__(self, stdout: str):
                self.stdout = stdout

        with patch.object(guard.subprocess, "run", side_effect=[Result("0, 3\n"), Result("912\n")]):
            sample = guard.query_gpu(0)
        self.assertEqual(sample, guard.GpuSample(0, 3, ("912",)))
        self.assertFalse(guard.IdleGate(5, 2048, 1).observe(0, sample)[0])

    def run_fake_wrapper(self, return_code: int) -> tuple[int, Path]:
        output = self.output / "run"
        config = self.output / "fit.yaml"
        config.write_text(f"output_dir: {output.as_posix()}\n", encoding="utf-8")
        trainer = self.output / "trainer"
        trainer.write_text("fake", encoding="utf-8")
        shutdown = self.output / "shutdown"
        shutdown.write_text("fake", encoding="utf-8")
        status = self.output / "status.json"

        class FakeProcess:
            pid = 1234
            returncode = return_code

            def __init__(inner_self, *args, **kwargs):
                if return_code == 0:
                    self.make_artifacts(output=output)

            def poll(inner_self):
                return return_code

        class ReadyGate:
            def __init__(inner_self, *args):
                pass

            def observe(inner_self, *args):
                return True, 600.0

        arguments = ["guard", "--config", str(config), "--trainer", str(trainer),
                     "--log-file", str(self.output / "train.log"),
                     "--status-file", str(status), "--arm-shutdown"]
        with (patch.object(sys, "argv", arguments),
              patch.object(guard.os, "geteuid", return_value=0, create=True),
              patch.object(guard.os, "access", return_value=True),
              patch.object(guard, "SHUTDOWN", shutdown),
              patch.object(guard.subprocess, "Popen", FakeProcess),
              patch.object(guard, "IdleGate", ReadyGate),
              patch.object(guard, "query_gpu", return_value=guard.GpuSample(0, 3, ())),
              patch.object(guard.subprocess, "run") as shutdown_call):
            result = guard.main()
            if return_code == 0:
                shutdown_call.assert_called_once_with(["/bin/sh", str(shutdown)],
                                                      check=True, timeout=30)
            else:
                shutdown_call.assert_not_called()
        return result, status

    def test_armed_shutdown_requires_successful_trainer_and_artifacts(self) -> None:
        result, status = self.run_fake_wrapper(return_code=0)
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(status.read_text(encoding="utf-8"))["phase"],
                         "shutdown_requested")

    def test_failed_trainer_never_calls_shutdown(self) -> None:
        result, status = self.run_fake_wrapper(return_code=1)
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(status.read_text(encoding="utf-8"))["phase"],
                         "training_failed")


if __name__ == "__main__":
    unittest.main()
