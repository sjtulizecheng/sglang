"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_scheduler(*, interval: int, require_mlp_sync: bool) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.require_mlp_sync = require_mlp_sync
    return scheduler


def _make_batch(*, local_extend: bool, global_extend: bool):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND if local_extend else ForwardMode.DECODE,
        is_extend_in_batch=global_extend,
    )


def _running_batch(*, empty=False, finished=False, prefill_only=False):
    return SimpleNamespace(
        reqs=[] if empty else [SimpleNamespace(finished=lambda: finished)],
        is_prefill_only=prefill_only,
    )


class TestPrefillDecodeInterval(unittest.TestCase):
    def test_disabled_interval_does_not_arm(self):
        scheduler = _make_scheduler(interval=0, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertFalse(scheduler._should_defer_prefill(_running_batch()))

    def test_non_dp_interval_uses_local_forward_mode(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertTrue(scheduler._should_defer_prefill(_running_batch()))
        self.assertTrue(scheduler._should_defer_prefill(_running_batch()))
        self.assertFalse(scheduler._should_defer_prefill(_running_batch()))

    def test_dp_interval_uses_globally_synchronized_extend_flag(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)

        # This rank is locally decoding, but another DP rank is prefilling.
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=True)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 2)
        self.assertTrue(scheduler._should_defer_prefill(_running_batch(empty=True)))

    def test_decode_batch_does_not_rearm_interval(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 1

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=False)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)

    def test_speculative_verification_does_not_rearm_local_interval(self):
        scheduler = _make_scheduler(interval=1, require_mlp_sync=False)
        scheduler._arm_prefill_decode_interval(
            SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        )
        self.assertFalse(scheduler._should_defer_prefill(_running_batch()))

    def test_non_dp_interval_expires_without_decode_work(self):
        for batch in [
            _running_batch(empty=True),
            _running_batch(finished=True),
            _running_batch(prefill_only=True),
        ]:
            with self.subTest(batch=batch):
                scheduler = _make_scheduler(interval=2, require_mlp_sync=False)
                scheduler._prefill_decode_interval_remaining = 2
                self.assertFalse(scheduler._should_defer_prefill(batch))
                self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)


if __name__ == "__main__":
    unittest.main()
