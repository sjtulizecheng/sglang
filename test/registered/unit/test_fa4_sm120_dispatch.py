"""SM120 must use MHA and respect the temporary KV materialization limit."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common import attention_backend_handler as handler
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)


class TestFA4SM120Dispatch(unittest.TestCase):
    def dispatch(self, seq_lens, capacity, mode=ForwardMode.EXTEND, sm120=True):
        batch = SimpleNamespace(
            forward_mode=mode,
            seq_lens_cpu=seq_lens,
            get_max_chunk_capacity=lambda: capacity,
        )
        with (
            patch.object(handler, "is_sm100_or_sm110_supported", return_value=False),
            patch.object(handler, "is_sm120_supported", return_value=sm120),
        ):
            return handler.handle_attention_fa4(None, batch)

    def test_combined_cached_and_new_kv_must_fit(self):
        # seq_lens includes the cached prefix. Using only the new token count
        # would select a one-shot path that exceeds its temporary buffer.
        self.assertEqual(
            self.dispatch([4096, 4096], 8192), AttnForwardMethod.MHA_ONE_SHOT
        )
        self.assertEqual(
            self.dispatch([4096, 4097], 8192), AttnForwardMethod.MHA_CHUNKED_KV
        )

    def test_unknown_lengths_and_other_paths_keep_chunked_kv(self):
        for mode, seq_lens, sm120 in [
            (ForwardMode.EXTEND, None, True),
            (ForwardMode.DECODE, [32], True),
            (ForwardMode.TARGET_VERIFY, [32], True),
            (ForwardMode.EXTEND, [32], False),
        ]:
            with self.subTest(mode=mode, seq_lens=seq_lens, sm120=sm120):
                self.assertEqual(
                    self.dispatch(seq_lens, 8192, mode, sm120),
                    AttnForwardMethod.MHA_CHUNKED_KV,
                )


if __name__ == "__main__":
    unittest.main()
