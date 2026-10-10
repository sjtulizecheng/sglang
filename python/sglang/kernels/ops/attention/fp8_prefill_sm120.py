# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental ragged FP8 attention for SM120, including cached prefixes.

Q/K/V are quantized per token and head. Both dot products use FP8 operands
with FP32 accumulation; softmax and normalization remain FP32. This does
not change the persistent KV cache representation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(
    X,
    Y,
    Scale,
    ROWS,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    R: tl.constexpr,
    HEAD_SCALES: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, D)
    x = tl.load(
        X + (row // H)[:, None] * S0 + (row % H)[:, None] * S1 + col[None, :] * S2,
        mask=row[:, None] < ROWS,
        other=0,
    ).to(tl.float32)
    if HEAD_SCALES:
        scale = tl.load(Scale + row % H)
    else:
        scale = tl.maximum(tl.max(tl.abs(x), axis=1) / 448.0, 1.0e-12)
    y = (x / scale[:, None]).to(tl.float8e4nv)
    tl.store(Y + row[:, None] * D + col[None, :], y, row[:, None] < ROWS)
    if not HEAD_SCALES:
        tl.store(Scale + row, scale, row < ROWS)


@triton.jit
def _attention(
    Q,
    K,
    V,
    SQ,
    SK,
    SV,
    CQ,
    CK,
    O,
    LSE,
    H: tl.constexpr,
    D: tl.constexpr,
    SM_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FP8_PV: tl.constexpr,
    V_HEAD_SCALE: tl.constexpr,
    FAST_ACCUM: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(CQ + batch)
    q_len = tl.load(CQ + batch + 1) - q_start
    k_start = tl.load(CK + batch)
    k_len = tl.load(CK + batch + 1) - k_start
    if tile * BM < q_len:
        m = tile * BM + tl.arange(0, BM)
        n = tl.arange(0, BN)
        d = tl.arange(0, D)
        q_row = (q_start + m) * H + head
        q = tl.load(Q + q_row[:, None] * D + d[None, :], m[:, None] < q_len, 0.0)
        sq = tl.load(SQ + q_row, m < q_len, 0)
        m_i = tl.full((BM,), -float("inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        end_n = k_len
        if CAUSAL:
            end_n = tl.minimum(end_n, (tile + 1) * BM + k_len - q_len)
        for start_n in range(0, end_n, BN):
            keys = start_n + n
            k_row = (k_start + keys) * H + head
            k = tl.load(K + k_row[None, :] * D + d[:, None], keys[None, :] < k_len, 0.0)
            sk = tl.load(SK + k_row, keys < k_len, 0)
            qk = tl.dot(q, k, max_num_imprecise_acc=D if FAST_ACCUM else 0).to(
                tl.float32
            )
            qk *= sq[:, None] * sk[None, :] * (SM_SCALE * 1.4426950408889634)
            allowed = keys[None, :] < k_len
            if CAUSAL:
                allowed = allowed & (keys[None, :] <= m[:, None] + k_len - q_len)
            qk = tl.where(allowed, qk, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            # Empty causal rows (Q longer than K) have zero output, -inf LSE.
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(qk - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc *= alpha[:, None]
            v = tl.load(V + k_row[:, None] * D + d[None, :], keys[:, None] < k_len, 0.0)
            if FP8_PV:
                if V_HEAD_SCALE:
                    p8 = (p * 448.0).to(tl.float8e4nv)
                    acc = tl.dot(
                        p8, v, acc, max_num_imprecise_acc=BN if FAST_ACCUM else 0
                    )
                else:
                    sv = tl.load(SV + k_row, keys < k_len, 0)
                    pv = p * sv[None, :]
                    sp = tl.maximum(tl.max(pv, axis=1) / 448.0, 1.0e-12)
                    p8 = (pv / sp[:, None]).to(tl.float8e4nv)
                    acc = tl.dot(
                        p8,
                        v,
                        acc / sp[:, None],
                        max_num_imprecise_acc=BN if FAST_ACCUM else 0,
                    )
                    acc *= sp[:, None]
            else:
                sv = tl.load(SV + k_row, keys < k_len, 0)
                v16 = (v.to(tl.float32) * sv[:, None]).to(tl.bfloat16)
                acc = tl.dot(p.to(tl.bfloat16), v16, acc)
            m_i = m_safe
        result = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        if V_HEAD_SCALE:
            result *= tl.load(SV + head) / 448.0
        tl.store(O + q_row[:, None] * D + d[None, :], result, m[:, None] < q_len)
        lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
        tl.store(LSE + q_row, lse, m < q_len)


def quantize_rows(x: torch.Tensor, head_scales=None):
    """Quantize [tokens, heads, dim] to E4M3 with per-row dequantization scales."""
    if x.ndim != 3 or x.shape[-1] not in (64, 128, 256):
        raise ValueError("Expected [tokens, heads, dim] with dim=64/128/256")
    y = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale = (
        torch.empty(x.shape[:2], dtype=torch.float32, device=x.device)
        if head_scales is None
        else head_scales
    )
    rows = x.shape[0] * x.shape[1]
    if rows:
        _quantize_rows[(triton.cdiv(rows, 4),)](
            x,
            y,
            scale,
            rows,
            x.shape[1],
            x.shape[2],
            *x.stride(),
            4,
            head_scales is not None,
        )
    return y, scale


def fp8_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    softmax_scale: float,
    *,
    causal: bool = True,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
    fp8_pv: bool = True,
    v_head_scale: bool = True,
    fast_accum: bool = False,
):
    """Return BF16 output and natural-log LSE; causal masks align to the right.

    Experimental API. Callers must validate numerical quality for their model.
    ``fp8_pv=False`` retains a BF16 second dot product for quality comparisons.
    """
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Expected FP16/BF16 source Q/K/V")
    if k.shape != v.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError("Only equal head counts and Q/K/V dimensions are supported")
    if cu_q.shape != cu_k.shape or cu_q.numel() < 2:
        raise ValueError("Q/K cumulative sequence lengths must have matching shapes")
    q8, sq = quantize_rows(q)
    k8, sk = quantize_rows(k)
    v_head_scale = v_head_scale and fp8_pv
    sv = (
        (v.abs().amax(dim=(0, 2)).float() / 448.0).clamp_min(1e-12)
        if v_head_scale
        else None
    )
    v8, sv = quantize_rows(v, sv)
    out = torch.empty_like(
        q, dtype=torch.bfloat16, memory_format=torch.contiguous_format
    )
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    if q.shape[0]:
        _attention[(triton.cdiv(max_q, block_m), q.shape[1], cu_q.numel() - 1)](
            q8,
            k8,
            v8,
            sq,
            sk,
            sv,
            cu_q,
            cu_k,
            out,
            lse,
            q.shape[1],
            q.shape[2],
            softmax_scale,
            causal,
            block_m,
            block_n,
            fp8_pv,
            v_head_scale,
            fast_accum,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out, lse
