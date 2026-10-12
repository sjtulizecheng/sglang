"""Opt-in BF16 operand / FP32 accumulator router GEMM for the GLM experiment."""

import json

import torch


def create_hook(config):
    from projection_tuning import create_hook as install_attention

    from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteGate

    hook = install_attention(config)
    previous = Glm4MoeLiteGate.forward
    checks = int(config.get("router_validation_calls", 3))
    topk = int(config.get("router_topk", 4))

    def forward(self, hidden_states):
        if (
            hidden_states.shape[0] < 32
            or hidden_states.dtype != torch.bfloat16
            or self.weight.dtype != torch.bfloat16
        ):
            return previous(self, hidden_states)
        # Checkpoint weights and model activations already are BF16. Preserve
        # FP32 logits/accumulation while avoiding FP32 casts and the SIMT GEMM.
        logits = torch.mm(hidden_states, self.weight.T, out_dtype=torch.float32)
        remaining = getattr(self, "_g80_router_checks", checks)
        if remaining:
            reference = previous(self, hidden_states)
            a = torch.sigmoid(reference) + self.e_score_correction_bias
            b = torch.sigmoid(logits) + self.e_score_correction_bias
            ids_a, ids_b = a.topk(topk, dim=-1).indices, b.topk(topk, dim=-1).indices
            same = torch.equal(ids_a.sort(-1).values, ids_b.sort(-1).values)
            print(
                "G80_ROUTER_CHECK",
                json.dumps(
                    {
                        "shape": list(hidden_states.shape),
                        "remaining": remaining,
                        "topk_set_equal": same,
                        "topk_ordered_equal": torch.equal(ids_a, ids_b),
                        "max_abs_difference": float((logits - reference).abs().max()),
                    }
                ),
                flush=True,
            )
            assert same, "BF16 Tensor Core router changed top-k set during warmup"
            self._g80_router_checks = remaining - 1
        return logits

    Glm4MoeLiteGate.forward = forward
    print("G80_ROUTER_TENSORCORE_INSTALLED", flush=True)
    return hook
