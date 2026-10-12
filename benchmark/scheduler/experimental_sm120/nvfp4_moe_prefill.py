"""Opt-in real-routing-screened NVFP4 MoE prefill tactics for SM120.

Requires the validated FlashInfer build; decode and other shapes use the original path.
"""

import json

import torch


def native_runner(fused_finalize=True):
    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module

    module = gen_cutlass_fused_moe_sm120_module(False).build_and_load()
    return module.init(
        torch.bfloat16,
        torch.int64,
        torch.bfloat16,
        False,
        False,
        False,
        False,
        fused_finalize,
    )


def native_forward(runner, args, output, tactics=(-1, -1), enable_pdl=True):
    runner.run_moe(
        output,
        args["input"],
        args["token_selected_experts"],
        args["token_final_scales"],
        args["fc1_expert_weights"],
        args.get("fc1_expert_biases"),
        args["fc2_expert_weights"],
        args.get("fc2_expert_biases"),
        args["quant_scales"],
        args.get("input_sf"),
        args.get("swiglu_alpha"),
        args.get("swiglu_beta"),
        args.get("swiglu_limit"),
        True,
        args.get("tp_size", 1),
        args.get("tp_rank", 0),
        args.get("ep_size", 1),
        args.get("ep_rank", 0),
        args.get("cluster_size", 1),
        args.get("cluster_rank", 0),
        args.get("enable_alltoall", False),
        False,
        list(tactics),
        enable_pdl,
        args["activation_type"],
    )
    return output


def install_tactics(config):
    """Late, prefill-only adapter for a screened pair; leave decode graphs intact."""
    import flashinfer

    from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as adapter

    if flashinfer.__version__ != "0.6.15.post1":
        raise RuntimeError("Experimental tactic IDs require FlashInfer 0.6.15.post1")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Experimental routed MoE tactics require SM120")
    original, activation_enum = adapter._flashinfer_cutlass_fused_moe()
    runner = native_runner()
    assert (runner.get_gemm1_tactic_count(), runner.get_gemm2_tactic_count()) == (
        20,
        40,
    )
    tactics = tuple(config.get("tactics", [12, 42]))
    assert tactics == (12, 42), "This adapter only accepts the bitwise screened pair"
    verify = bool(config.get("validate_each_layer", True))
    verified = {}
    shapes = set()

    def forward(**kwargs):
        x = kwargs["input"]
        w1, w2 = kwargs["fc1_expert_weights"], kwargs["fc2_expert_weights"]
        eligible = (
            x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[1] == 2048
            and 1024 <= x.shape[0] <= 16384
            and w1.dtype == w2.dtype == torch.int64
            and tuple(w1.shape) == (64, 3072, 128)
            and tuple(w2.shape) == (64, 2048, 96)
            and tuple(kwargs["token_selected_experts"].shape) == (x.shape[0], 4)
            and kwargs["output_dtype"] == torch.bfloat16
            and kwargs.get("input_sf") is None
            and int(kwargs["activation_type"]) == 3
            and not kwargs.get("enable_alltoall", False)
            and kwargs.get("tp_size", 1) == kwargs.get("ep_size", 1) == 1
        )
        if not eligible:
            return original(**kwargs)
        output = kwargs["output"]
        key = w1.data_ptr()
        reference = None
        if verify and key not in verified:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Prefill tactic validation must happen outside graph capture"
                )
            reference = original(**kwargs)[0].clone()
        native_forward(runner, kwargs, output, tactics)
        if reference is not None:
            same = torch.equal(output, reference)
            print(
                "G80_REAL_MOE_TACTIC_CHECK",
                json.dumps(
                    {
                        "validated_layers": len(verified) + 1,
                        "m": x.shape[0],
                        "bitwise_equal": same,
                        "tactics": tactics,
                    }
                ),
                flush=True,
            )
            if not same:
                raise RuntimeError(
                    "Native routed tactic did not match original output exactly"
                )
            verified[key] = (w1, w2)
        if x.shape[0] not in shapes and len(shapes) < 12:
            shapes.add(x.shape[0])
            print("G80_REAL_MOE_TACTIC_EXECUTED", x.shape[0], tactics, flush=True)
        return [output]

    adapter._flashinfer_cutlass_fused_moe = lambda: (forward, activation_enum)
    print("G80_REAL_MOE_TACTICS_INSTALLED", tactics, flush=True)
    return lambda module, inputs, output: None
