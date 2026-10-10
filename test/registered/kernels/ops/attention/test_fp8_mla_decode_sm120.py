"""Native FP8 dense MLA decode: indexed pages and CUDA graph replay."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "Requires SM120/SM121",
)
class TestFP8MLADecodeSM120(unittest.TestCase):
    def reference(self, q, cache, ptr, indices):
        rows = []
        for i in range(q.shape[0]):
            kv = cache[indices[ptr[i] : ptr[i + 1]].long(), 0].float()
            if kv.shape[0] == 0:
                rows.append(torch.zeros_like(q[i, :, :512], dtype=torch.float32))
            else:
                scores = q[i].float() @ kv.T / 16
                rows.append(scores.softmax(-1) @ kv[:, :512])
        return torch.stack(rows)

    def check_output(self, actual, expected):
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(
            ((actual.float() - expected).norm() / expected.norm()).item(), 0.06
        )

    @torch.inference_mode()
    def test_variable_lengths_empty_rows_and_padded_heads(self):
        from sglang.kernels.ops.attention.fp8_mla_decode_sm120 import fp8_mla_decode

        torch.manual_seed(2026)
        for heads in (3, 20, 33):
            with self.subTest(heads=heads):
                q = torch.randn((5, heads, 576), device="cuda", dtype=torch.bfloat16)
                cache = torch.randn((2401, 1, 576), device="cuda").to(
                    torch.float8_e4m3fn
                )
                ptr = torch.tensor(
                    [0, 0, 1, 38, 295, 1320], device="cuda", dtype=torch.int32
                )
                indices = torch.randperm(2401, device="cuda", dtype=torch.int32)[:1320]
                expected = self.reference(q, cache, ptr, indices)
                actual = fp8_mla_decode(
                    q[:, :, :512], q[:, :, 512:], cache, ptr, indices, 1 / 16
                )
                self.check_output(actual, expected)
                torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))

    @torch.inference_mode()
    def test_graph_replay_reads_updated_indices_and_lengths(self):
        from sglang.kernels.ops.attention.fp8_mla_decode_sm120 import fp8_mla_decode

        torch.manual_seed(73)
        q = torch.randn((2, 20, 576), device="cuda", dtype=torch.bfloat16)
        cache = torch.randn((2048, 1, 576), device="cuda").to(torch.float8_e4m3fn)
        ptr = torch.tensor([0, 17, 110], device="cuda", dtype=torch.int32)
        indices = torch.randperm(2048, device="cuda", dtype=torch.int32)

        def run():
            return fp8_mla_decode(
                q[:, :, :512], q[:, :, 512:], cache, ptr, indices, 1 / 16
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        for lengths in ([0, 39, 813], [0, 0, 9]):
            ptr.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            indices.copy_(torch.randperm(2048, device="cuda", dtype=torch.int32))
            q.normal_()
            graph.replay()
            self.check_output(out, self.reference(q, cache, ptr, indices))


if __name__ == "__main__":
    unittest.main()
