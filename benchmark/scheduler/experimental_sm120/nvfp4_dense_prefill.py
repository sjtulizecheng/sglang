"""Local opt-in NVFP4 dense prefill experiment; quality is not yet accepted."""

import json

import torch


def create_hook(config):
    from flashinfer import mm_fp4, nvfp4_quantize
    from tensorcore_router import create_hook as install_router

    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    hook = install_router(config)
    previous = Fp8LinearMethod.apply
    shapes = {
        tuple(s)
        for s in config.get(
            "dense_shapes",
            [[8960, 512], [5120, 768], [2048, 5120], [3072, 2048], [2048, 1536]],
        )
    }
    minimum = int(config.get("dense_minimum_tokens", 1024))
    global_scale = float(config.get("dense_activation_global_scale", 16))
    scale_overrides = {
        tuple(item["shape"]): float(item["scale"])
        for item in config.get("dense_scale_overrides", [])
    }
    validations = int(config.get("dense_validation_calls", 1))
    dispatch = None
    if config.get("dense_cached_dispatch", False):
        from nvfp4_dispatch import CachedCutlassFp4

        dispatch = CachedCutlassFp4(config["dense_autotune_cache"])
        print("G80_DENSE_CACHED_DISPATCH_INSTALLED", flush=True)

    def apply(self, layer, x, bias=None):
        if (
            not isinstance(x, torch.Tensor)
            or x.dtype != torch.bfloat16
            or x.ndim != 2
            or x.shape[0] < minimum
            or not self.block_quant
            or self.use_mxfp8
            or tuple(layer.weight.shape) not in shapes
            or list(self.weight_block_size) != [128, 128]
            or layer.weight.dtype != torch.float8_e4m3fn
        ):
            return previous(self, layer, x, bias)
        cache = getattr(layer, "_g80_dense_nvfp4", None)
        if cache is None:
            w, s = layer.weight, layer.weight_scale_inv
            wf = (
                w.float()
                * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[
                    : w.shape[0], : w.shape[1]
                ]
            ).bfloat16()
            wg = 2688.0 / wf.float().abs().max().clamp_min(1e-12)
            w4, ws = nvfp4_quantize(wf, wg)
            actual_scale = scale_overrides.get(tuple(w.shape), global_scale)
            ag = torch.full((), actual_scale, device=w.device, dtype=torch.float32)
            cache = {
                "w": w4,
                "s": ws,
                "ag": ag,
                "alpha": (wg * ag).reciprocal(),
                "source_weight": w,
                "source_scale": s,
                "checks": validations,
                "calls": 0,
            }
            layer._g80_dense_nvfp4 = cache
            print(
                "G80_DENSE_NVFP4_PREPARED",
                json.dumps(
                    {
                        "shape": list(w.shape),
                        "prefix": getattr(layer, "prefix", ""),
                        "activation_scale": actual_scale,
                    }
                ),
                flush=True,
            )
        assert (
            cache["source_weight"] is layer.weight
            and cache["source_scale"] is layer.weight_scale_inv
        )
        a, sa = nvfp4_quantize(x.contiguous(), cache["ag"])
        if dispatch is None:
            out = mm_fp4(
                a,
                cache["w"].T,
                sa,
                cache["s"].T,
                alpha=cache["alpha"],
                out_dtype=torch.bfloat16,
                backend="cutlass",
            )
        else:
            out = dispatch(a, cache["w"], sa, cache["s"], cache["alpha"])
            if cache["checks"]:
                public = mm_fp4(
                    a,
                    cache["w"].T,
                    sa,
                    cache["s"].T,
                    alpha=cache["alpha"],
                    out_dtype=torch.bfloat16,
                    backend="cutlass",
                )
                assert torch.equal(
                    out, public
                ), "Cached dispatch changed tuned CUTLASS output"
                print(
                    "G80_DENSE_DISPATCH_EXACT",
                    json.dumps(
                        {"shape": list(layer.weight.shape), "tokens": x.shape[0]}
                    ),
                    flush=True,
                )
        if bias is not None:
            out.add_(bias)
        if cache["checks"]:
            reference = previous(self, layer, x, bias)
            finite = bool(torch.isfinite(out).all())
            print(
                "G80_DENSE_NVFP4_CHECK",
                json.dumps(
                    {
                        "shape": list(layer.weight.shape),
                        "tokens": x.shape[0],
                        "finite": finite,
                        "activation_amax": float(x.float().abs().max()),
                        "relative_l2": float(
                            (out.float() - reference.float()).norm()
                            / reference.float().norm().clamp_min(1e-12)
                        ),
                    }
                ),
                flush=True,
            )
            assert finite, "NVFP4 dense prefill produced non-finite output"
            cache["checks"] -= 1
        cache["calls"] += 1
        if cache["calls"] == 4:
            print(
                "G80_DENSE_NVFP4_ACTIVE",
                json.dumps(
                    {"shape": list(layer.weight.shape), "calls": cache["calls"]}
                ),
                flush=True,
            )
        return out

    Fp8LinearMethod.apply = apply
    print(
        "G80_DENSE_NVFP4_INSTALLED",
        json.dumps(
            {
                "shapes": sorted(shapes),
                "minimum_tokens": minimum,
                "activation_scale": global_scale,
            }
        ),
        flush=True,
    )
    return hook
