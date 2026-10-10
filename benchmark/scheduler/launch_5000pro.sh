#!/usr/bin/env bash
# Reproduce the user's baseline and isolate each scheduler change.
set -euo pipefail

profile=${1:-e2e}
if (( $# )); then shift; fi
: "${MODEL:?Set MODEL to the local model directory or Hugging Face model ID}"
max_running_requests=6

export SGLANG_OPT_MOE_QUANT_ONCE=1
export SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL:-/opt/trtllm_gen_moe_cubin_pool}
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=256
export SGLANG_GLM_FUSED_GATE_TOPK=1
export SGLANG_MOE_ROUND0_SINGLETON=1
export SGLANG_GLM_QKVA_BF16=1

case "$profile" in
  e2e-slo)
    max_running_requests=16
    knobs=(--schedule-policy fcfs --chunked-prefill-size 16384 --prefill-decode-interval 0 --pd-slo-ms 5000)
    ;;
  e2e)
    max_running_requests=16
    knobs=(--schedule-policy fcfs --chunked-prefill-size 16384 --prefill-decode-interval 0)
    ;;
  baseline) knobs=(--schedule-policy fcfs --chunked-prefill-size 16384) ;;
  interval1) knobs=(--schedule-policy fcfs --chunked-prefill-size 16384 --prefill-decode-interval 1) ;;
  fcfs4096) knobs=(--schedule-policy fcfs --chunked-prefill-size 4096 --prefill-decode-interval 1) ;;
  spf4096) knobs=(--schedule-policy shortest-prefill-first --chunked-prefill-size 4096 --prefill-decode-interval 1) ;;
  hrrn4096) knobs=(--schedule-policy hrrn --chunked-prefill-size 4096 --prefill-decode-interval 1) ;;
  mixed4096) knobs=(--schedule-policy shortest-prefill-first --chunked-prefill-size 4096 --enable-mixed-chunk) ;;
  delayer4096) knobs=(--schedule-policy shortest-prefill-first --chunked-prefill-size 4096 --enable-prefill-delayer --prefill-delayer-max-delay-passes 1 --prefill-delayer-max-delay-ms 10) ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

exec sglang serve \
  --model-path "$MODEL" --served-model-name chat --trust-remote-code \
  --tp-size 1 --host 0.0.0.0 --port "${PORT:-31010}" --mem-fraction-static 0.9 \
  --attention-backend flashinfer --disable-shared-experts-fusion \
  --max-running-requests "$max_running_requests" --decode-log-interval 1 \
  "${knobs[@]}" "$@"
