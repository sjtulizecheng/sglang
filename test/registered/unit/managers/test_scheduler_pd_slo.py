"""Exercise the controller through actual scheduler and radix-cache paths."""

import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.pd_slo_controller import PDSLOController
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def req(rid, length=3000, generated=0):
    r = Req(rid, "", array("q", range(length)), SamplingParams(max_new_tokens=128))
    r.output_ids.extend([7] * generated)
    r.time_stats.created_time = time.perf_counter() - 1
    return r


class TestSchedulerPDSLO(unittest.TestCase):
    def test_initialization_does_not_require_later_dllm_initialization(self):
        s = Scheduler.__new__(Scheduler)
        s.is_generation = True
        s.model_config = SimpleNamespace(is_multimodal=False)
        s.ps = SimpleNamespace(pp_size=1)
        schedule = SimpleNamespace(
            chunked_prefill_size=16384,
            prefill_decode_interval=0,
            pd_slo_ms=5000,
            enable_mixed_chunk=False,
            enable_dynamic_chunking=False,
        )
        execution = SimpleNamespace(dllm=SimpleNamespace(dllm_algorithm=None))
        with patch(
            "sglang.srt.managers.scheduler.get_schedule", return_value=schedule
        ), patch(
            "sglang.srt.managers.scheduler.get_exec", return_value=execution
        ), patch(
            "sglang.srt.managers.scheduler.is_cuda", return_value=True
        ), patch(
            "sglang.srt.managers.scheduler.get_resolved_model_impl", return_value=None
        ):
            s.init_chunked_prefill()
        self.assertEqual(s.pd_slo_controller.slo, 5)

    def test_real_cache_match_changes_phase_without_allocating_request_slots(self):
        s = Scheduler.__new__(Scheduler)
        s.pd_slo_controller = PDSLOController(5)
        for size in range(1, 6):
            s.pd_slo_controller.observe_forward(
                prefill=False, tokens=size, batch_size=size, seconds=0.01
            )
        s.pd_slo_controller.observe_forward(
            prefill=True, tokens=3000, batch_size=1, seconds=0.12
        )
        s.tree_cache = RadixCache.create_simulated()
        s.page_size = 1
        s.chunked_req = None
        s.chunked_prefill_size = s.max_prefill_tokens = 16384
        s.server_args = SimpleNamespace(prefill_max_requests=None)
        s.get_num_allocatable_reqs = lambda running_bs: 12 - running_bs
        pending = req("pending")
        pending.time_stats.created_time = time.perf_counter()
        s.waiting_queue = [pending]
        running = SimpleNamespace(
            reqs=[req(str(i), length=32, generated=126) for i in range(4)],
            is_prefill_only=False,
        )
        self.assertTrue(s._should_defer_prefill(running))
        self.assertIsNone(pending.req_pool_idx)
        s.tree_cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(2999))), value=torch.arange(2999)
            )
        )
        self.assertFalse(s._should_defer_prefill(running))
        self.assertEqual(pending.num_matched_prefix_tokens, 2999)
        self.assertIsNone(pending.req_pool_idx)

    def test_only_successful_completions_are_learned_once(self):
        s = Scheduler.__new__(Scheduler)
        s.pd_slo_controller = PDSLOController(5)
        finished = req("finished", generated=128)
        finished.finished_reason = FINISH_LENGTH(128)
        aborted = req("aborted", generated=3)
        aborted.finished_reason = FINISH_ABORT()
        batch = SimpleNamespace(pd_slo_timing=None, reqs=[finished, aborted])
        s._pd_slo_observe_result(batch)
        s._pd_slo_observe_result(batch)
        self.assertEqual(list(s.pd_slo_controller.output_lengths), [(128, 128)])


if __name__ == "__main__":
    unittest.main()
