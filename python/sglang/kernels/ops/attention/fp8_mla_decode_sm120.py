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
"""Experimental dense MLA decode: FP8 cache and native FP8 dot products.

Reads only indexed cache slots. Supports unit-scale E4M3 latent+RoPE storage
(512+64), not the block-scaled sparse-MLA page format. Softmax and split-K
merging use FP32; Q is dynamically quantized per head and component.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_partials(
    QN,
    QR,
    KV,
    Indptr,
    Indices,
    Part,
    Lse,
    H: tl.constexpr,
    QNS0: tl.constexpr,
    QNS1: tl.constexpr,
    QRS0: tl.constexpr,
    QRS1: tl.constexpr,
    KVS0: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1) * BH + tl.arange(0, BH)
    split = tl.program_id(2)
    dn = tl.arange(0, 512)
    dr = tl.arange(0, 64)
    n = tl.arange(0, BN)
    qn = tl.load(
        QN + batch * QNS0 + head[:, None] * QNS1 + dn[None, :], head[:, None] < H, 0
    ).to(tl.float32)
    qr = tl.load(
        QR + batch * QRS0 + head[:, None] * QRS1 + dr[None, :], head[:, None] < H, 0
    ).to(tl.float32)
    sn = tl.maximum(tl.max(tl.abs(qn), 1) / 448.0, 1.0e-12)
    sr = tl.maximum(tl.max(tl.abs(qr), 1) / 448.0, 1.0e-12)
    qn8 = (qn / sn[:, None]).to(tl.float8e4nv)
    qr8 = (qr / sr[:, None]).to(tl.float8e4nv)
    begin = tl.load(Indptr + batch)
    length = tl.load(Indptr + batch + 1) - begin
    chunk = tl.cdiv(length, SPLITS * BN) * BN
    start = split * chunk
    stop = tl.minimum(start + chunk, length)
    mi = tl.full((BH,), -float("inf"), tl.float32)
    li = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, 512), tl.float32)
    for offset in range(start, stop, BN):
        pos = offset + n
        index = tl.load(Indices + begin + pos, pos < stop, 0)
        kn = tl.load(KV + index[None, :] * KVS0 + dn[:, None], pos[None, :] < stop, 0.0)
        kr = tl.load(
            KV + index[None, :] * KVS0 + 512 + dr[:, None], pos[None, :] < stop, 0.0
        )
        scores = tl.dot(qn8, kn) * sn[:, None] + tl.dot(qr8, kr) * sr[:, None]
        scores *= SCALE * 1.4426950408889634
        scores = tl.where(pos[None, :] < stop, scores, -float("inf"))
        mn = tl.maximum(mi, tl.max(scores, 1))
        alpha = tl.exp2(mi - mn)
        p = tl.exp2(scores - mn[:, None])
        li = li * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        acc = tl.dot((p * 448.0).to(tl.float8e4nv), tl.trans(kn), acc)
        mi = mn
    result = acc / (tl.where(li > 0, li, 1.0) * 448.0)[:, None]
    base = (batch * H + head) * SPLITS + split
    tl.store(Part + base[:, None] * 512 + dn[None, :], result, head[:, None] < H)
    tl.store(Lse + base, mi + tl.log2(li), head < H)


@triton.jit
def _merge(Part, Lse, O, SPLITS: tl.constexpr, BS: tl.constexpr):
    row = tl.program_id(0)
    split = tl.arange(0, BS)
    d = tl.arange(0, 512)
    ls = tl.load(Lse + row * SPLITS + split, split < SPLITS, -float("inf"))
    mx = tl.max(ls, 0)
    mx = tl.where(mx == -float("inf"), 0.0, mx)
    weight = tl.exp2(ls - mx)
    den = tl.sum(weight, 0)
    partial = tl.load(
        Part + (row * SPLITS + split[:, None]) * 512 + d[None, :],
        split[:, None] < SPLITS,
        0,
    )
    out = tl.sum(partial * weight[:, None], 0) / tl.where(den > 0, den, 1.0)
    tl.store(O + row * 512 + d, out)


def fp8_mla_decode(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    *,
    num_splits: int = 16,
    block_k: int = 64,
    num_warps: int = 8,
):
    """Decode one token per request from indexed unit-scale FP8 MLA cache."""
    if q_nope.ndim != 3 or q_nope.shape[-1] != 512:
        raise ValueError("Expected Q-nope [batch, heads, 512]")
    if q_rope.shape != (*q_nope.shape[:2], 64):
        raise ValueError("Expected Q-rope [batch, heads, 64]")
    if (
        kv_cache.dtype != torch.float8_e4m3fn
        or kv_cache.ndim != 3
        or kv_cache.shape[1:] != (1, 576)
    ):
        raise ValueError("Expected unit-scale E4M3 cache with 512+64 dimensions")
    tensors = (q_nope, q_rope, kv_cache, kv_indptr, kv_indices)
    if not q_nope.is_cuda or any(t.device != q_nope.device for t in tensors):
        raise ValueError("All inputs must be on the same CUDA device")
    if (
        q_nope.dtype not in (torch.bfloat16, torch.float16)
        or q_rope.dtype != q_nope.dtype
    ):
        raise ValueError("Q components must have matching BF16 or FP16 dtype")
    if (
        num_splits < 1
        or num_splits > 64
        or block_k not in (32, 64, 128)
        or num_warps not in (4, 8)
    ):
        raise ValueError("Invalid split count, KV block size, or warp count")
    b, h, _ = q_nope.shape
    if h < 1 or kv_indptr.numel() < b + 1:
        raise ValueError("Expected positive head count and batch+1 KV offsets")
    for offsets in (kv_indptr, kv_indices):
        if (
            offsets.ndim != 1
            or offsets.stride(0) != 1
            or offsets.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "KV metadata must be contiguous one-dimensional integer tensors"
            )
    if q_nope.stride(-1) != 1 or q_rope.stride(-1) != 1 or kv_cache.stride(-1) != 1:
        raise ValueError("The innermost dimension must be contiguous")
    out = torch.empty_like(q_nope, memory_format=torch.contiguous_format)
    if b == 0:
        return out
    partial = torch.empty(
        (b, h, num_splits, 512), device=q_nope.device, dtype=torch.float32
    )
    lse = torch.empty((b, h, num_splits), device=q_nope.device, dtype=torch.float32)
    _decode_partials[(b, triton.cdiv(h, 32), num_splits)](
        q_nope,
        q_rope,
        kv_cache,
        kv_indptr,
        kv_indices,
        partial,
        lse,
        h,
        *q_nope.stride()[:2],
        *q_rope.stride()[:2],
        kv_cache.stride(0),
        softmax_scale,
        num_splits,
        32,
        block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    _merge[(b * h,)](
        partial, lse, out, num_splits, triton.next_power_of_2(num_splits), num_warps=4
    )
    return out
