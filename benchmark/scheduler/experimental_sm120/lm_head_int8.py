"""Experimental INT8 LM-head storage with BF16 Tensor Core arithmetic.

Install before CUDA Graph capture. Static BF16 source weights are retained.
Validation evidence and quality limits are recorded in this directory README.
"""

import json

import torch
import triton
import triton.language as tl


@triton.jit
def _lm_head_int8_weight_only_kernel(
    X, W, S, Y, N, M, K, BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BK)
    ms = tl.arange(0, BM)
    acc = tl.zeros((BN, BM), tl.float32)
    for k in range(0, K, BK):
        kk = k + cols
        w = tl.load(
            W + rows[:, None] * K + kk[None, :],
            (rows[:, None] < N) & (kk[None, :] < K),
            0.0,
        ).to(tl.bfloat16)
        x = tl.load(
            X + ms[:, None] * K + kk[None, :], (ms[:, None] < M) & (kk[None, :] < K), 0
        )
        acc = tl.dot(w, tl.trans(x), acc)
    scale = tl.load(S + rows, rows < N, 0)
    logits = acc * scale[:, None]
    tl.store(
        Y + ms[:, None] * N + rows[None, :],
        tl.trans(logits).to(tl.bfloat16),
        (ms[:, None] < M) & (rows[None, :] < N),
    )


def forward(x, w, s, *, bn=64, bk=128, warps=4, stages=3):
    m, k = x.shape
    n = w.shape[0]
    assert x.dtype == torch.bfloat16 and w.dtype == torch.int8
    assert (
        m <= 16
        and w.shape[1] == k
        and x.is_contiguous()
        and w.is_contiguous()
        and s.is_contiguous()
    )
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _lm_head_int8_weight_only_kernel[(triton.cdiv(n, bn),)](
        x, w, s, y, n, m, k, bn, bk, 16, num_warps=warps, num_stages=stages
    )
    return y


def install_lm_head(config):
    """Install before graph capture; the dense prefill hook stays separate."""
    from sglang.srt.layers.logits_processor import LogitsProcessor

    previous = LogitsProcessor._compute_lm_head
    validations = int(config.get("lm_head_validation_calls", 1))

    @torch.no_grad()
    def compute(self, hidden_states, lm_head, embedding_bias=None):
        weight = getattr(lm_head, "weight", None)
        if (
            weight is None
            or tuple(weight.shape) != (154880, 2048)
            or weight.dtype != torch.bfloat16
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 2
            or not 1 <= hidden_states.shape[0] <= 16
            or self.use_fp32_lm_head
            or self.rl_on_policy_target is not None
            or hasattr(lm_head, "set_lora")
            or embedding_bias is not None
        ):
            return previous(self, hidden_states, lm_head, embedding_bias)
        cache = getattr(lm_head, "_g80_int8_weight_only", None)
        if cache is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("INT8 LM-head must be initialized in graph warmup")
            w8 = torch.empty_like(weight, dtype=torch.int8)
            scales = torch.empty(
                (weight.shape[0], 1), device=weight.device, dtype=torch.float32
            )
            for offset in range(0, weight.shape[0], 8192):
                chunk = weight[offset : offset + 8192].float()
                scale = chunk.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
                scales[offset : offset + 8192].copy_(scale)
                w8[offset : offset + 8192].copy_(
                    (chunk / scale).round().clamp(-127, 127).to(torch.int8)
                )
            cache = {
                "weight": w8,
                "scale": scales,
                "source": weight,
                "checks": validations,
                "captured_batches": set(),
            }
            lm_head._g80_int8_weight_only = cache
            print(
                "G80_LM_HEAD_INT8_W8A16_PREPARED",
                json.dumps(
                    {
                        "shape": list(weight.shape),
                        "activation_dtype": "bf16",
                        "weight_dtype": "int8",
                    }
                ),
                flush=True,
            )
        assert cache["source"] is weight, "LM-head weight changed after quantization"
        output = forward(hidden_states.contiguous(), cache["weight"], cache["scale"])
        if (
            torch.cuda.is_current_stream_capturing()
            and hidden_states.shape[0] not in cache["captured_batches"]
        ):
            cache["captured_batches"].add(hidden_states.shape[0])
            print("G80_LM_HEAD_INT8_GRAPH_TRACE", hidden_states.shape[0], flush=True)
        if cache["checks"] and not torch.cuda.is_current_stream_capturing():
            reference = previous(self, hidden_states, lm_head, embedding_bias)
            finite = bool(torch.isfinite(output).all())
            print(
                "G80_LM_HEAD_INT8_W8A16_CHECK",
                json.dumps(
                    {
                        "batch": hidden_states.shape[0],
                        "finite": finite,
                        "relative_l2": float(
                            (output.float() - reference.float()).norm()
                            / reference.float().norm().clamp_min(1e-12)
                        ),
                    }
                ),
                flush=True,
            )
            assert finite, "INT8 LM-head returned non-finite logits"
            cache["checks"] -= 1
        return output

    LogitsProcessor._compute_lm_head = compute
    print("G80_LM_HEAD_INT8_W8A16_INSTALLED", flush=True)
    return lambda module, inputs, output: None
