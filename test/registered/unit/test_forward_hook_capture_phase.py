"""Kernel-installing hooks must run before capture when explicitly requested."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.hook_manager import split_forward_hooks_by_capture_phase
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as setup
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestForwardHookCapturePhase(unittest.TestCase):
    def capture(self, specs):
        events = []
        runner = SimpleNamespace(
            server_args=SimpleNamespace(forward_hooks=specs, enable_symm_mem=False),
            model=object(),
            is_draft_worker=False,
            device="cuda",
            forward_stream=None,
            canary_manager=None,
        )

        def register(model, hooks):
            events.extend(spec["name"] for spec in hooks)

        def graph(phase, **kwargs):
            events.append(phase)
            return setup.GraphCapture(
                runner=None, memory_phase=phase, memory_usage_gb=0, capture_time=0
            )

        with (
            patch.object(
                setup.GraphSharedOutput,
                "create_for_model_runner",
                return_value=object(),
            ),
            patch.object(
                setup,
                "EagerRunner",
                side_effect=lambda _: events.append("warmup") or object(),
            ),
            patch.object(setup, "register_forward_hooks", side_effect=register),
            patch.object(
                setup,
                "capture_prefill_graph",
                side_effect=lambda **kw: graph("prefill", **kw),
            ),
            patch.object(
                setup,
                "capture_decode_graph",
                side_effect=lambda **kw: graph("decode", **kw),
            ),
            patch.object(setup, "prealloc_symmetric_memory_pool"),
        ):
            setup.capture_cuda_graphs(model_runner=runner)
        return events

    def test_ordinary_hooks_remain_after_capture(self):
        self.assertEqual(
            self.capture([{"name": "ordinary"}]),
            ["warmup", "prefill", "decode", "ordinary"],
        )

    def test_opted_in_kernel_factory_precedes_warmup_and_capture(self):
        specs = [
            {"name": "ordinary"},
            {"name": "kernel1", "capture_phase": "before_graph"},
            {"name": "kernel2", "capture_phase": "before_graph"},
            {"name": "late", "capture_phase": "after_graph"},
        ]
        self.assertEqual(
            self.capture(specs),
            ["kernel1", "kernel2", "warmup", "prefill", "decode", "ordinary", "late"],
        )

    def test_invalid_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "capture_phase"):
            self.capture([{"name": "typo", "capture_phase": "before"}])

    def test_no_hooks(self):
        self.assertEqual(split_forward_hooks_by_capture_phase(None), ([], []))
        self.assertEqual(self.capture([]), ["warmup", "prefill", "decode"])


if __name__ == "__main__":
    unittest.main()
