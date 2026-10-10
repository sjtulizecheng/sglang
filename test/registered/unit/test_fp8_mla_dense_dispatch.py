"""Dense FP8 MLA opt-in must bypass whole-pool conversion only on decode."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFP8MLADenseDispatch(unittest.TestCase):
    def fixture(self, enabled=True, spec_info=None):
        backend = FlashInferMLAAttnBackend.__new__(FlashInferMLAAttnBackend)
        backend.use_sm120_fp8_decode = enabled
        cache = torch.zeros((16, 1, 576), dtype=torch.float8_e4m3fn)
        backend.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _: cache)
        backend.kv_indptr = torch.tensor([0, 3, 7])
        out = torch.zeros((2, 20, 512), dtype=torch.bfloat16)
        wrapper = SimpleNamespace(
            run=Mock(return_value=out),
            _kv_indptr_buf=torch.tensor([0, 1, 2]),
            _kv_indices_buf=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        )
        backend.forward_metadata = SimpleNamespace(decode_wrapper=wrapper)
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=20, v_head_dim=512, head_dim=576, scaling=1 / 16
        )
        batch = SimpleNamespace(
            out_cache_loc=None, forward_mode=ForwardMode.DECODE, spec_info=spec_info
        )
        q = torch.zeros((2, 20, 576), dtype=torch.bfloat16)
        return backend, wrapper, layer, batch, q, out

    def test_native_decode_uses_updated_offsets_and_wrapper_indices(self):
        backend, wrapper, layer, batch, q, out = self.fixture()
        # An FP8 object whose .to() fails detects accidental whole-pool casts.
        cache = Mock(dtype=torch.float8_e4m3fn, ndim=3, shape=(16, 1, 576))
        cache.to.side_effect = AssertionError("Unexpected whole-pool conversion")
        backend.token_to_kv_pool.get_key_buffer = lambda _: cache
        with patch(
            "sglang.kernels.ops.attention.fp8_mla_decode_sm120.fp8_mla_decode",
            return_value=out,
        ) as native:
            result = backend.forward_decode(q, None, None, layer, batch)
        self.assertEqual(result.shape, (2, 20 * 512))
        args = native.call_args.args
        self.assertIs(args[2], cache)
        # fast_mla_decode_plan leaves the wrapper offsets at capture values.
        self.assertIs(args[3], backend.kv_indptr)
        self.assertIsNot(args[3], wrapper._kv_indptr_buf)
        self.assertIs(args[4], wrapper._kv_indices_buf)
        wrapper.run.assert_not_called()

    def test_disabled_and_speculative_paths_use_existing_backend(self):
        for enabled, spec_info in [(False, None), (True, object())]:
            with self.subTest(enabled=enabled, speculative=spec_info is not None):
                backend, wrapper, layer, batch, q, out = self.fixture(
                    enabled, spec_info
                )
                with (
                    patch(
                        "sglang.srt.layers.attention.flashinfer_mla_backend.get_parallel",
                        return_value=SimpleNamespace(dcp_enabled=False),
                    ),
                    patch(
                        "sglang.kernels.ops.attention.fp8_mla_decode_sm120.fp8_mla_decode"
                    ) as native,
                ):
                    result = backend.forward_decode(q, None, None, layer, batch)
                self.assertEqual(result.shape, (2, 20 * 512))
                wrapper.run.assert_called_once()
                self.assertEqual(wrapper.run.call_args.args[2].dtype, torch.bfloat16)
                native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
