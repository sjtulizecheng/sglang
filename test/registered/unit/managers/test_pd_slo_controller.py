"""Deadline decisions and online estimates without a GPU or model downloads."""

import pickle
import unittest

from sglang.srt.managers.pd_slo_controller import PDSLOController, SLORequest
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def controller(slo=5):
    c = PDSLOController(slo)
    for n in range(1, 9):
        c.observe_forward(prefill=False, tokens=n, batch_size=n, seconds=0.01)
    c.observe_forward(prefill=True, tokens=3000, batch_size=1, seconds=0.12)
    return c


class TestPDSLOController(unittest.TestCase):
    def test_finish_short_decodes_before_expensive_prefill(self):
        c = controller()
        self.assertTrue(
            c.should_decode([SLORequest(1, 2)] * 4, [SLORequest(0, 80)], 3000, 10)
        )

    def test_admit_cheap_cached_prefill_when_decodes_have_far_to_go(self):
        c = controller()
        self.assertFalse(
            c.should_decode([SLORequest(1, 80)] * 4, [SLORequest(0, 80)], 1, 10)
        )

    def test_slo_changes_phase_for_same_requests_and_costs(self):
        running = [SLORequest(1, 2)]
        waiting = [SLORequest(4.7, 80)]
        self.assertTrue(controller(10).should_decode(running, waiting, 3000, 10))
        self.assertFalse(controller(5).should_decode(running, waiting, 3000, 10))

    def test_decode_slice_cannot_starve_prefill(self):
        c = controller()
        running, waiting = [SLORequest(1, 2)] * 4, [SLORequest(0, 80)]
        self.assertTrue(c.should_decode(running, waiting, 3000, 10))
        self.assertFalse(c.should_decode(running, waiting, 3000, 10.51))
        self.assertEqual(c.counts["bounded_prefill"], 1)

    def test_empty_decode_or_prefill_queue_never_defers(self):
        c = controller()
        for running, waiting in [([], [SLORequest(0, 80)]), ([SLORequest(0, 80)], [])]:
            c.defer_started = 1
            self.assertFalse(c.should_decode(running, waiting, 3000, 2))
            self.assertIsNone(c.defer_started)

    def test_prefill_admission_does_not_stack_another_long_decode_slice(self):
        c = controller()
        self.assertFalse(
            c.should_decode([SLORequest(1, 10)] * 3, [SLORequest(0, 80)], 3000, 10)
        )

    def test_prefill_interruption_reserves_decode_progress(self):
        c = controller()
        running, waiting = [SLORequest(1, 80)] * 4, [SLORequest(0, 80)]
        self.assertFalse(c.should_decode(running, waiting, 1, 10))
        c.observe_forward(prefill=True, tokens=3000, batch_size=1, seconds=0.12)
        self.assertTrue(c.should_decode(running, waiting, 1, 10.12))
        for _ in range(12):
            c.observe_forward(prefill=False, tokens=4, batch_size=4, seconds=0.01)
        self.assertFalse(c.should_decode(running, waiting, 1, 10.24))

    def test_nearer_deadline_reserves_more_decode_time(self):
        waiting = [SLORequest(0, 80)]
        relaxed, urgent = controller(), controller()
        relaxed.should_decode([SLORequest(1, 80)] * 4, waiting, 1, 10)
        urgent.should_decode([SLORequest(4, 80)] * 4, waiting, 1, 10)
        for c in (relaxed, urgent):
            c.observe_forward(prefill=True, tokens=3000, batch_size=1, seconds=0.12)
        self.assertGreater(urgent.decode_credit, relaxed.decode_credit)
        self.assertLessEqual(urgent.decode_credit, 0.5)

    def test_decode_credit_cannot_starve_prefill_and_resets_when_idle(self):
        c = controller()
        running, waiting = [SLORequest(4, 80)] * 4, [SLORequest(0, 80)]
        c.should_decode(running, waiting, 1, 10)
        c.observe_forward(prefill=True, tokens=3000, batch_size=1, seconds=1)
        self.assertTrue(c.should_decode(running, waiting, 1, 11))
        self.assertFalse(c.should_decode(running, waiting, 1, 11.51))
        self.assertEqual(c.decode_credit, 0)
        c.observe_forward(prefill=True, tokens=3000, batch_size=1, seconds=0.12)
        c.should_decode([], waiting, 1, 12)
        self.assertEqual(c.decode_credit, 0)
        self.assertEqual(c.target_decode_share, 0)

    def test_remaining_length_conditions_on_survival_and_honors_limit(self):
        c = controller()
        for _ in range(16):
            c.observe_completion(20, 128)
            c.observe_completion(120, 128)
        self.assertEqual(c.remaining_tokens(100, 128), 20)
        self.assertEqual(c.remaining_tokens(127, 128), 1)
        self.assertEqual(c.remaining_tokens(10, 16), 6)

    def test_cost_model_learns_prefill_and_batch_dependent_decode(self):
        c = PDSLOController(5)
        for tokens in [1000, 3000, 5000]:
            c.observe_forward(
                prefill=True, tokens=tokens, batch_size=1, seconds=0.01 + tokens / 10000
            )
        self.assertAlmostEqual(c.predict_prefill(2000), 0.21)
        c.observe_forward(prefill=False, tokens=2, batch_size=2, seconds=0.01)
        c.observe_forward(prefill=False, tokens=4, batch_size=4, seconds=0.02)
        self.assertAlmostEqual(c.predict_decode(3), 0.015)

    def test_unspecified_output_cap_learns_across_context_lengths(self):
        c = PDSLOController(1.5)
        # The scheduler clips an unspecified max_tokens to context - input.
        # None of these caps repeats, although all outputs naturally end early.
        for index in range(32):
            c.observe_completion(20, 100000 - index)
        self.assertAlmostEqual(c.remaining_tokens(5, 99900), 15)
        self.assertEqual(c.remaining_tokens(5, 12), 7)
        self.assertEqual(c.stats()["max_completion_limit_samples"], 1)

    def test_pooled_estimate_does_not_treat_truncation_as_eos(self):
        c = PDSLOController(1.5)
        for index in range(8):
            c.observe_completion(4, 4)
            c.observe_completion(12, 100 + index)
        # The short capped outputs are censored; all observed natural endings
        # occur at 12, so pooling the raw lengths into a mean of 8 is wrong.
        self.assertAlmostEqual(c.remaining_tokens(0, 128), 12)
        self.assertAlmostEqual(c.remaining_tokens(6, 128), 6)

    def test_pooled_estimate_keeps_unobserved_tail_conservative(self):
        c = PDSLOController(1.5)
        for index in range(8):
            c.observe_completion(10 + index, 10 + index)
        self.assertEqual(c.remaining_tokens(0, 128), 128)

    def test_decode_projection_shrinks_batch_after_completion(self):
        c = controller()
        costs = {1: 0.01, 2: 0.02}
        requests = [SLORequest(0, 2), SLORequest(0, 10)]
        loss, elapsed, survivors = c._decode_plan(requests, costs=costs)
        self.assertAlmostEqual(elapsed, 0.12)
        self.assertAlmostEqual(loss, 0.04 + 0.12)
        self.assertEqual(survivors, [])
        loss, elapsed, survivors = c._decode_plan(requests, steps=5, costs=costs)
        self.assertAlmostEqual(loss, 0.04)
        self.assertAlmostEqual(elapsed, 0.07)
        self.assertEqual(survivors, [SLORequest(0, 5)])

    def test_ingress_timestamp_survives_ipc_without_metrics(self):
        stats = APIServerReqTimeStats()
        stats.set_created_time()
        received = pickle.loads(pickle.dumps(stats))
        self.assertGreater(received.created_time, 0)
        self.assertAlmostEqual(received.created_time, stats.created_time, places=3)


if __name__ == "__main__":
    unittest.main()
