"""Summarize completed P/D SLO trials without copying request bodies."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(root):
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        directory = summary_path.parent
        row = json.loads(summary_path.read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        before = json.loads((directory / "server-before.json").read_text())
        after = json.loads((directory / "server-after.json").read_text())
        first = (
            before["internal_states"][0].get("pd_slo_stats", {}).get("decisions", {})
        )
        last = after["internal_states"][0].get("pd_slo_stats", {}).get("decisions", {})
        row.update(
            {
                "pd_slo_ms": before["pd_slo_ms"],
                "max_running_requests": before["max_running_requests"],
                "chunked_prefill_size": before["chunked_prefill_size"],
                "arrival_seed": metadata["arrival_seed"],
                "sampling_seed_active": before.get(
                    "enable_deterministic_inference", False
                ),
                "case_sha256": metadata["selected_dataset_sha256"],
                "source_sha256": metadata["source_sha256"],
                "decisions": {
                    key: value - first.get(key, 0) for key, value in last.items()
                },
            }
        )
        rows.append(row)
    return rows


def compare(root, rows, baseline_prefixes, candidate_prefixes):
    """Pool raw latencies for explicitly selected repeated trial sequences."""
    groups = {}
    for name, prefixes in [
        ("baseline", baseline_prefixes),
        ("candidate", candidate_prefixes),
    ]:
        groups[name] = [
            row
            for row in rows
            if any(row["label"].startswith(prefix + "-") for prefix in prefixes)
        ]
        for prefix in prefixes:
            selected = [
                row for row in groups[name] if row["label"].startswith(prefix + "-")
            ]
            if sorted((r["cache"], r["arrival_seed"]) for r in selected) != [
                ("cold", 43),
                ("warm", 43),
                ("warm", 44),
            ]:
                raise ValueError(
                    f"{prefix} must contain a complete cold/warm/warm suite"
                )
    selected = groups["baseline"] + groups["candidate"]
    for key in [
        "case_sha256",
        "source_sha256",
        "max_running_requests",
        "chunked_prefill_size",
        "sampling_seed_active",
    ]:
        if any(row[key] != selected[0][key] for row in selected):
            raise ValueError(f"Comparison differs in {key}")

    result = {
        "baseline_prefixes": baseline_prefixes,
        "candidate_prefixes": candidate_prefixes,
        "sampling_seed_active": selected[0]["sampling_seed_active"],
        "percentile_method": "linear interpolation over pooled per-request latencies",
        "cache_states": {},
    }
    for cache in ["cold", "warm"]:
        arms = {}
        for name, group in groups.items():
            trials = [row for row in group if row["cache"] == cache]
            latencies = []
            output_tokens = 0
            for row in trials:
                raw = json.loads((root / row["label"] / "result.json").read_text())
                latencies.extend(raw["latencies_s"])
                output_tokens += sum(raw["output_lens"])
            ordered = sorted(latencies)
            position = (len(ordered) - 1) * 0.95
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (
                position - lower
            )
            arms[name] = {
                "trials": [row["label"] for row in trials],
                "completed": sum(row["completed"] for row in trials),
                "mean_e2e_ms": statistics.mean(latencies) * 1000,
                "p95_e2e_ms": p95 * 1000,
                "output_tokens": output_tokens,
                "all_trials_meet_slo": all(row["slo_met"] for row in trials),
            }
        arms["candidate_mean_change_pct"] = 100 * (
            arms["candidate"]["mean_e2e_ms"] / arms["baseline"]["mean_e2e_ms"] - 1
        )
        result["cache_states"][cache] = arms
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--baseline-prefixes", nargs="+")
    parser.add_argument("--candidate-prefixes", nargs="+")
    args = parser.parse_args()
    if bool(args.baseline_prefixes) != bool(args.candidate_prefixes):
        parser.error("Supply both baseline and candidate prefixes")
    rows = summarize(args.report_dir)
    report = (
        compare(args.report_dir, rows, args.baseline_prefixes, args.candidate_prefixes)
        if args.baseline_prefixes
        else rows
    )
    print(json.dumps(report, indent=2))
