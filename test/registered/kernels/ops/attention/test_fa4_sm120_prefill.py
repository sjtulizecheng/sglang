"""SM120 FA4 prefill: real GLM head size, ragged sequences and cached prefixes."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "Requires SM120/SM121",
)
class TestFA4SM120Prefill(unittest.TestCase):
    @torch.inference_mode()
    def test_excess_tiles_ignore_poisoned_sequence_metadata(self):
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            flash_attn_varlen_func,
        )

        q = torch.zeros((256, 20, 256), device="cuda", dtype=torch.bfloat16)
        v = torch.empty_like(q)
        v[:128].fill_(0.25)
        v[128:].fill_(0.75)
        # The varlen grid overestimates the number of tiles for aligned lengths.
        # Put a large sentinel immediately beyond the logical metadata so an
        # invalid tile cannot accidentally see benign allocator padding.
        cu = torch.tensor([0, 128, 256, 1 << 28], device="cuda", dtype=torch.int32)[:3]
        out = flash_attn_varlen_func(
            q,
            q,
            v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=128,
            max_seqlen_k=128,
            causal=True,
        )
        torch.testing.assert_close(out, v, atol=0.01, rtol=0.01)

    @torch.inference_mode()
    def test_varlen_prefill_and_prefix_attention(self):
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            flash_attn_varlen_func,
        )

        generator = torch.Generator(device="cuda").manual_seed(42)
        for dim in [128, 256]:
            for causal in [False, True]:
                with self.subTest(dim=dim, causal=causal):
                    q_lengths, k_lengths = [65, 129], [97, 193]
                    q = torch.randn(
                        (sum(q_lengths), 20, dim),
                        device="cuda",
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                    k, v = [
                        torch.randn(
                            (sum(k_lengths), 20, dim),
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=generator,
                        )
                        for _ in range(2)
                    ]
                    cu_q = torch.tensor([0, 65, 194], device="cuda", dtype=torch.int32)
                    cu_k = torch.tensor([0, 97, 290], device="cuda", dtype=torch.int32)
                    out, lse = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q=cu_q,
                        cu_seqlens_k=cu_k,
                        max_seqlen_q=max(q_lengths),
                        max_seqlen_k=max(k_lengths),
                        causal=causal,
                        return_softmax_lse=True,
                    )
                    ref_outputs, ref_lses = [], []
                    q_start = k_start = 0
                    for nq, nk in zip(q_lengths, k_lengths):
                        q_part = q[q_start : q_start + nq].float().transpose(0, 1)
                        k_part = k[k_start : k_start + nk].float().transpose(0, 1)
                        v_part = v[k_start : k_start + nk].float().transpose(0, 1)
                        scores = q_part @ k_part.transpose(-1, -2) / dim**0.5
                        if causal:
                            # FA causal masks align queries at the end of KV,
                            # which is essential when the cached prefix is nonempty.
                            allowed = torch.arange(nk, device="cuda")[None, :] <= (
                                torch.arange(nq, device="cuda")[:, None] + nk - nq
                            )
                            scores.masked_fill_(~allowed, -torch.inf)
                        ref_outputs.append(
                            (scores.softmax(-1) @ v_part).transpose(0, 1)
                        )
                        ref_lses.append(scores.logsumexp(-1))
                        q_start += nq
                        k_start += nk
                    ref = torch.cat(ref_outputs)
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertLess(
                        ((out.float() - ref).norm() / ref.norm()).item(), 0.01
                    )
                    torch.testing.assert_close(
                        lse, torch.cat(ref_lses, dim=1), atol=0.01, rtol=0.001
                    )


if __name__ == "__main__":
    unittest.main()
