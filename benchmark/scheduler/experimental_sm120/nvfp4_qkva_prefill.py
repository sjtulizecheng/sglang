"""Local, explicitly enabled NVFP4 QKV-A prefill candidate for quality tests."""

import json

import torch


def create_hook(config):
    from flashinfer import nvfp4_quantize
    from nvfp4_dense_prefill import create_hook as install_dense
    from nvfp4_dispatch import CachedCutlassFp4

    from sglang.srt.layers.linear import ReplicatedLinear
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
    from sglang.srt.runtime_context import get_exec

    hook = install_dense(config)
    previous = ReplicatedLinear.forward
    dispatch = CachedCutlassFp4(config["dense_autotune_cache"])
    activation_scale = float(config.get("qkva_activation_global_scale", 16))
    validations = int(config.get("qkva_validation_calls", 1))

    def forward(self, x):
        weight = getattr(self, "weight", None)
        scale = getattr(self, "weight_scale_inv", None)
        if (
            x.ndim != 2
            or x.shape[0] < 1024
            or x.shape[1] != 2048
            or x.dtype != torch.bfloat16
            or weight is None
            or scale is None
            or weight.shape != (1344, 2048)
            or scale.shape != (11, 16)
            or weight.dtype != torch.float8_e4m3fn
            or getattr(self, "bias", None) is not None
            or getattr(self, "set_lora", False)
            or get_exec().deterministic.enable_deterministic_inference
        ):
            return previous(self, x)
        cache = getattr(self, "_g80_qkva_nvfp4", None)
        if cache is None:
            bf = block_quant_dequant(weight, scale, [128, 128], torch.bfloat16)
            padded = torch.zeros(
                (1408, 2048), device=weight.device, dtype=torch.bfloat16
            )
            padded[:1344].copy_(bf)
            wg = 2688.0 / padded.float().abs().max().clamp_min(1e-12)
            w4, sw = nvfp4_quantize(padded, wg)
            ag = torch.full(
                (), activation_scale, device=weight.device, dtype=torch.float32
            )
            cache = {
                "w": w4,
                "s": sw,
                "ag": ag,
                "alpha": (wg * ag).reciprocal(),
                "weight": weight,
                "scale": scale,
                "checks": validations,
            }
            self._g80_qkva_nvfp4 = cache
            print("G80_QKVA_NVFP4_PREPARED", getattr(self, "prefix", ""), flush=True)
        assert cache["weight"] is weight and cache["scale"] is scale
        a, sa = nvfp4_quantize(x.contiguous(), cache["ag"])
        out = dispatch(a, cache["w"], sa, cache["s"], cache["alpha"])[:, :1344]
        if cache["checks"]:
            reference = previous(self, x)[0]
            assert torch.isfinite(out).all()
            print(
                "G80_QKVA_NVFP4_CHECK",
                json.dumps(
                    {
                        "tokens": x.shape[0],
                        "activation_amax": float(x.float().abs().max()),
                        "relative_l2": float(
                            (out.float() - reference.float()).norm()
                            / reference.float().norm()
                        ),
                    }
                ),
                flush=True,
            )
            cache["checks"] -= 1
        return out, None

    ReplicatedLinear.forward = forward
    print("G80_QKVA_NVFP4_INSTALLED", activation_scale, flush=True)
    return hook
