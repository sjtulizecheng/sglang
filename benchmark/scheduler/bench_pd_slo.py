"""One controlled CHAT trial for an on/off comparison of --pd-slo-ms.

Uses the user's HTTP runner unchanged. The optional per-request sampling seed
is identical in both arms; input, output limits and sampling settings are kept.
This branch only uses sampling seeds with deterministic inference enabled;
the original backend configuration still requires comparing output lengths.
Run a cold trial before warm trials on a freshly started server.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--cache", choices=["cold", "warm"], required=True)
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=Path("/sgl-workspace/hyg-llm-benchmark-cases"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:31010")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--arrival-seed", type=int, default=43)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--request-rate", type=float, default=3.0)
    args = parser.parse_args()
    if not math.isfinite(args.request_rate) or args.request_rate <= 0:
        parser.error("--request-rate must be finite and positive")
    sys.path.insert(0, str(args.bench_root / "scripts"))
    import auto_qps
    import http_bench

    directory = args.bench_root / "bench-report-auto" / "pd-slo-5000pro" / args.label
    directory.mkdir(parents=True, exist_ok=False)
    cases, original_hash = auto_qps.load_cases(
        args.bench_root / "CHAT.jsonl", disable_stream=True
    )
    selected = auto_qps.prepare_cases(cases, args.num_prompts, args.sample_seed, False)
    for index, case in enumerate(selected):
        case.setdefault("seed", args.sampling_seed + index)
    dataset = directory / "cases.jsonl"
    dataset.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in selected)
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, method="GET"):
        with opener.open(
            urllib.request.Request(args.base_url + path, method=method), timeout=30
        ) as response:
            data = response.read()
            if not data:
                return None
            if response.headers.get_content_type() == "application/json":
                return json.loads(data)
            return data.decode()

    loads = request("/v1/loads")["loads"]
    if any(load["num_running_reqs"] or load["num_waiting_reqs"] for load in loads):
        raise RuntimeError("Run the comparison only after the server has drained")
    if args.cache == "cold":
        request("/flush_cache", "POST")
    (directory / "loads-before.json").write_text(
        json.dumps(request("/v1/loads"), indent=2)
    )
    info = request("/server_info")
    (directory / "server-before.json").write_text(json.dumps(info, indent=2))
    root = Path(__file__).resolve().parents[2]
    names = [
        "managers/pd_slo_controller.py",
        "managers/scheduler.py",
        "managers/schedule_batch.py",
        "server_args.py",
        "observability/req_time_stats.py",
    ]
    sources = {name: (root / "python/sglang/srt" / name).read_text() for name in names}
    metadata = {
        **vars(args),
        "sampling_seed_active": bool(info.get("enable_deterministic_inference", False)),
        "original_dataset_sha256": original_hash,
        "selected_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in sources.items()
        },
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    (directory / "sources.json").write_text(json.dumps(sources, indent=2))
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,power.draw,clocks.sm",
            "--format=csv",
        ],
        stdout=(directory / "gpu.txt").open("w"),
        check=True,
    )
    output = directory / "result.json"
    http_bench.run(
        SimpleNamespace(
            dataset_path=dataset,
            base_url=args.base_url,
            model="chat",
            num_prompts=args.num_prompts,
            request_rate=args.request_rate,
            seed=args.arrival_seed,
            warmup_requests=1,
            timeout=120.0,
            max_concurrency=None,
            output_file=output,
        )
    )
    after = request("/server_info")
    (directory / "server-after.json").write_text(json.dumps(after, indent=2))
    (directory / "loads-after.json").write_text(
        json.dumps(request("/v1/loads"), indent=2)
    )
    result = json.loads(output.read_text())
    summary = {
        "label": args.label,
        "cache": args.cache,
        "completed": result["completed"],
        "mean_e2e_ms": statistics.mean(result["latencies_s"]) * 1000,
        "p95_e2e_ms": result["p95_e2e_latency_ms"],
        "output_tokens": sum(result["output_lens"]),
        "request_rate": args.request_rate,
        "achieved_qps": result["request_throughput"],
        "duration_s": result["duration"],
        "output_tokens_per_second": sum(result["output_lens"]) / result["duration"],
        "slo_met": result["completed"] >= 0.99 * args.num_prompts
        and result["p95_e2e_latency_ms"] <= 5000,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
