#!/usr/bin/env bash
# Default: cold cache (except one warmup), then warm cache if the first trial passes.
# CACHE_STATE=warm keeps the cache populated by a previous run of the same cases.
# Run only on an idle benchmark server. HTTP auto_qps ignores --flush-cache.
set -euo pipefail

label=${1:?Usage: bash bench_5000pro.sh PROFILE_LABEL}
bench_root=${BENCH_ROOT:-/sgl-workspace/hyg-llm-benchmark-cases}
base_url=${BASE_URL:-http://127.0.0.1:31010}
report_dir=${REPORT_DIR:-$bench_root/bench-report-auto/scheduler-5000pro}/$label
mkdir -p "$report_dir"

case "${CACHE_STATE:-cold}" in
  cold) curl --fail --silent --show-error --max-time 15 -X POST "$base_url/flush_cache" ;;
  warm) ;;
  *) echo "CACHE_STATE must be cold or warm" >&2; exit 2 ;;
esac
printf '%s\n' "${CACHE_STATE:-cold}" > "$report_dir/cache-state.txt"
curl --fail --silent --show-error --max-time 15 "$base_url/get_server_info" > "$report_dir/server-info.json"
git rev-parse HEAD > "$report_dir/git-head.txt"
git diff --binary > "$report_dir/worktree.patch"
nvidia-smi > "$report_dir/gpu.txt"
python3 - "$report_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path('python/sglang/srt')
files = ['managers/scheduler.py', 'managers/schedule_policy.py',
         'managers/pd_slo_controller.py', 'managers/schedule_batch.py',
         'server_args.py', 'observability/req_time_stats.py']
snapshots = {name: (source / name).read_text() for name in files}
Path(sys.argv[1], 'scheduler-source.json').write_text(json.dumps({
    'sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in snapshots.items()},
    'sources': snapshots,
}, indent=2))
PY
bench_exit=0
python3 "$bench_root/scripts/auto_qps.py" \
  --runner http --base-url "$base_url" --served-model chat \
  --cases "$bench_root/CHAT.jsonl" --report-dir "$report_dir" \
  --start-qps 3 --min-qps 3 --max-qps 3 \
  --slo-p95-e2e-ms 5000 --min-success-rate 0.99 \
  --num-prompts 200 --repeats 2 --disable-stream --seed 42 || bench_exit=$?
curl --fail --silent --show-error --max-time 15 "$base_url/get_server_info" > "$report_dir/server-info-after.json"
exit "$bench_exit"
