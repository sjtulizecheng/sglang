"""Exercise HRRN accounting in the real scheduler entry and dispatch paths."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _BeforeForward(Exception):
    pass


class TestSchedulerHRRN(unittest.TestCase):
    def test_only_prefill_tokens_advance_aging(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.forward_ct = 0
        scheduler.processed_tokens_counter = 0
        scheduler.scripted_scheduler_hook = SimpleNamespace(
            on_run_batch=Mock(side_effect=_BeforeForward)
        )
        # Stop at the dispatch hook, before GPU work. Decode retains the old
        # extend count; speculative verification is not new prefill work.
        for mode, expected in [
            (ForwardMode.EXTEND, 1024),
            (ForwardMode.DECODE, 1024),
            (ForwardMode.TARGET_VERIFY, 1024),
            (ForwardMode.IDLE, 1024),
            (ForwardMode.MIXED, 2045),
        ]:
            with self.subTest(mode=mode):
                batch = SimpleNamespace(
                    forward_mode=mode,
                    extend_num_tokens=1024,
                    decoding_reqs=[object()] * 3,
                )
                with self.assertRaises(_BeforeForward):
                    scheduler.run_batch(batch)
                self.assertEqual(scheduler.processed_tokens_counter, expected)

    def test_arrival_and_requeue_snapshot_current_counter(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.waiting_queue = []
        scheduler._set_or_validate_priority = Mock(return_value=True)
        scheduler._abort_on_queued_limit = Mock(return_value=False)
        scheduler._prefetch_kvcache = Mock()
        req = SimpleNamespace(time_stats=Mock())
        for count, retracted in [(1024, False), (4096, True)]:
            scheduler.processed_tokens_counter = count
            scheduler._add_request_to_queue(req, is_retracted=retracted)
            self.assertIs(scheduler.waiting_queue.pop(), req)
            self.assertEqual(req.arrival_processed_tokens, count)


if __name__ == "__main__":
    unittest.main()
