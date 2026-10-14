"""Opt-in GLM prefill adapter for FlashInfer's SM120 FMHA-v2 TMA kernel.

Reuse the original fused FP8 preparation (per-head, 64-token Q/K/V scales),
then run packed variable-length attention without padding Q to the KV length.
Compile the source checkout using the installed FlashInfer JIT to preserve the
older Python package used by the MoE experiments.
"""

import functools
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch


@functools.lru_cache(None)
def load_kernel(source_root="/sgl-workspace/flashinfer"):
    from flashinfer.jit import current_compilation_context
    from flashinfer.jit import env as jit_env
    from flashinfer.jit import gen_jit_spec

    root = Path(source_root).resolve()
    codegen = root / "flashinfer/jit/attention/fmha_v2"
    csrc = root / "csrc"
    inputs = sorted((csrc / "fmha_v2").rglob("*")) + [
        csrc / "trtllm_fmha_v2_binding.cu",
        csrc / "tvm_ffi_utils.h",
        codegen / "generate_kernels.py",
        codegen / "generator_utils.py",
    ]
    digest = hashlib.sha256()
    for path in inputs:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    name = "glm_sm120_fmha_v2_" + digest.hexdigest()[:16]
    package_name = "_" + name
    package = types.ModuleType(package_name)
    package.__path__ = [str(codegen)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        package_name + ".generate_kernels", codegen / "generate_kernels.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generated = jit_env.FLASHINFER_GEN_SRC_DIR / name
    generator.enumerate_kernels(csrc / "fmha_v2", generated)
    # Keep the checkout's binding and all its referenced variants unchanged.
    kernel_names = [
        "bf16_64_128_S_q_k_v_192x128",
        "e4m3_fp32_64_64_S_q_k_v_64x64_output_bf16",
        "e4m3_fp32_64_64_S_q_k_v_128x128_output_bf16",
        "e4m3_fp32_64_64_S_q_k_v_192x128_output_bf16",
        "e4m3_fp32_64_64_S_q_k_v_192x128_output_fp16",
        "e4m3_fp32_64_64_S_q_k_v_256x256_output_bf16",
        "e4m3_fp32_64_64_S_q_k_v_256x256_output_fp16",
        "e4m3_fp32_64_64_S_q_k_v_192x128",
    ]
    sources = [
        generated / "generated" / f"fmha_v2_flash_attention_{n}_sm120.cu"
        for n in kernel_names
    ] + [csrc / "trtllm_fmha_v2_binding.cu"]
    flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    ) + ["-Wno-deprecated-gpu-targets"]
    module = gen_jit_spec(
        name,
        sources,
        extra_cuda_cflags=flags,
        extra_include_paths=[csrc, csrc / "fmha_v2"],
    ).build_and_load()
    print("SM120_FMHA_V2_KERNEL_LOADED", name, str(root), flush=True)
    return module


def prepare(q, k_nope, k_rope, v, cu_q, cu_k, max_q, max_k, softmax_scale):
    """Use exactly the same two preparation kernels as the original fused path."""
    from materialized_fp8_quantize import prepare as prepare_fp8

    q8, k8, v8, scales = prepare_fp8(
        q, k_nope, k_rope, v, cu_q, cu_k, max_q, max_k, 64, 64
    )
    return q8, k8, v8, scales, cu_q, cu_k, max_q, max_k, softmax_scale


def run_packed(module, packed, out=None, lse=None, causal=True):
    q8, k8, v8, scales, cu_q, cu_k, max_q, max_k, softmax_scale = packed
    if out is None:
        out = torch.empty(q8.shape, device=q8.device, dtype=torch.bfloat16)
    module.run_glm(
        q8,
        k8,
        v8,
        out,
        cu_q,
        cu_k,
        *scales,
        lse,
        max_q,
        max_k,
        softmax_scale,
        causal,
    )
    return out


def install_hook(config):
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This GLM FMHA-v2 adapter is validated on SM120 only")
    module = load_kernel(config.get("source_root", "/sgl-workspace/flashinfer"))
    previous_prepare = HybridAttnBackend.prepare_prefill_qkv
    previous_extend = FlashAttentionBackend.forward_extend
    minimum = int(config.get("minimum_query_tokens", 1024))
    max_kv_tokens = int(config.get("max_kv_tokens", 32768))
    calls = 0

    def prepare_qkv(self, *, q, q_pe, kv_a, k_pe, positions, layer, forward_batch):
        backend = self.prefill_backend
        qlens = [int(n) for n in forward_batch.extend_seq_lens_cpu]
        klens = [int(n) for n in forward_batch.seq_lens_cpu]
        eligible = (
            isinstance(backend, FlashAttentionBackend)
            and backend.fa_impl_ver == 4
            and not layer.use_dsa
            and layer.num_local_heads == 20
            and (layer.qk_nope_head_dim, layer.qk_rope_head_dim, layer.v_head_dim)
            == (192, 64, 256)
            and q.dtype == torch.bfloat16
            and forward_batch.mha_one_shot
            and q.shape[0] >= minimum
            and q.shape[0] == forward_batch.extend_num_tokens
            and forward_batch.spec_info is None
            and qlens
            and klens
            and all(0 < nq <= nk for nq, nk in zip(qlens, klens))
            and len(qlens) == len(klens)
            and sum(klens) <= max_kv_tokens
        )
        if not eligible:
            return previous_prepare(
                self,
                q=q,
                q_pe=q_pe,
                kv_a=kv_a,
                k_pe=k_pe,
                positions=positions,
                layer=layer,
                forward_batch=forward_batch,
            )
        if layer.rotary_emb is not None:
            q_pe, k_pe = layer.rotary_emb(positions, q_pe, k_pe)
        q[..., 192:] = q_pe
        layer._set_mla_kv_buffer(None, kv_a, k_pe, forward_batch)
        if any(forward_batch.extend_prefix_lens_cpu):
            kv_a, k_pe = layer._get_mla_kv_buffer(
                forward_batch.fetch_mha_one_shot_kv_indices(), q.dtype, forward_batch
            )
        kv = layer.kv_b_proj(kv_a)[0].view(-1, 20, 448)
        metadata = backend.forward_metadata
        packed = prepare(
            q,
            kv[..., :192],
            k_pe,
            kv[..., 192:],
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
            metadata.max_seq_len_q,
            metadata.max_seq_len_k,
            layer.scaling,
        )
        forward_batch._sm120_fmha_v2 = (layer.layer_id, q.shape[0], packed)
        return packed[0], packed[1], packed[2]

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        nonlocal calls
        pending = getattr(forward_batch, "_sm120_fmha_v2", None)
        if pending is None:
            return previous_extend(
                self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        layer_id, nq, packed = pending
        assert layer_id == layer.layer_id and not save_kv_cache
        assert forward_batch.mha_one_shot and not forward_batch.mha_return_lse
        assert forward_batch.attn_attend_prefix_cache is False
        target = getattr(forward_batch, "_attn_output", None)
        if target is not None:
            target = target.view(nq, 20, 256)
        out = run_packed(module, packed, out=target)
        del forward_batch._sm120_fmha_v2
        if layer.layer_id == 0:
            calls += 1
            if calls <= 3 or calls % 50 == 0:
                print(
                    "SM120_FMHA_V2_PREFILL",
                    json.dumps(
                        {
                            "calls": calls,
                            "query_tokens": nq,
                            "padded_query_tokens": 0,
                            "scale_block_tokens": 64,
                            "kv_tokens": int(sum(forward_batch.seq_lens_cpu)),
                        }
                    ),
                    flush=True,
                )
        return out

    HybridAttnBackend.prepare_prefill_qkv = prepare_qkv
    FlashAttentionBackend.forward_extend = forward_extend
    print(
        "SM120_FMHA_V2_PREFILL_INSTALLED",
        minimum,
        max_kv_tokens,
        "packed_block64",
        flush=True,
    )

    def hook(module, inputs, output):
        pass

    return hook


if __name__ == "__main__":
    load_kernel(
        os.environ.get("SGLANG_SM120_FMHA_V2_SOURCE", "/sgl-workspace/flashinfer")
    )
