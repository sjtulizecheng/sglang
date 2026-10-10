"""Summarize auto_qps HTTP trials, including mean E2E omitted by its SLO table."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(root):
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        summary = json.loads(summary_path.read_text())
        for trial in summary.get("history", []):
            result_path = trial.get("result_file")
            if not result_path or not Path(result_path).is_file():
                continue
            result = json.loads(Path(result_path).read_text())
            errors = result["errors"]
            latencies = result["latencies_s"]
            successful = [v * 1000 for v, err in zip(latencies, errors) if not err]
            rows.append(
                {
                    "run": str(summary_path.parent.relative_to(root)),
                    "repeat": trial["repeat"],
                    "completed": result["completed"],
                    "attempted": len(errors),
                    "mean_e2e_ms": statistics.mean(successful) if successful else None,
                    "mean_all_attempts_ms": statistics.mean(latencies) * 1000,
                    "p95_e2e_ms": result["p95_e2e_latency_ms"],
                    "achieved_qps": result["request_throughput"],
                    "output_tokens": sum(result["output_lens"]),
                    "max_client_delay_ms": max(result["client_delays_s"]) * 1000,
                    "slo_met": trial.get("met", False),
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.report_dir), indent=2, ensure_ascii=False))
