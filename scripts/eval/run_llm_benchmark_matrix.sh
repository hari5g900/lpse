#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT="checkpoints/nljepahybrid.pt"
RES="artifacts/results"
mkdir -p "$RES"

export KUBECONFIG="${KUBECONFIG:-$PWD/config}"
if [ -f artifacts/node_scrape_map.json ]; then
  export NODE_SCRAPE_URL_MAP="$(python3 -c "import json;print(json.dumps(json.load(open('artifacts/node_scrape_map.json'))))")"
fi

QWEN_URL="${QWEN_URL:-http://localhost:8001/v1/chat/completions}"
QWEN_MODEL="Qwen/Qwen3-4B-Instruct-2507-FP8"
NEMO_URL="${NEMO_URL:-http://10.68.184.17:8000/v1/chat/completions}"
NEMO_MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
EXCLUDE_NODE_IP="${EXCLUDE_NODE_IP:-10.68.184.21}"

COMMON=(--checkpoint "$CKPT" --steps 20 --questions-per-step 5 --max-questions 100 --sample-interval-s 2.0)

run_cond() {
  local model=$1 cond=$2 mode=$3; shift 3
  local base="${RES}/${model}_${cond}_${mode}"
  echo "== $model/$cond/$mode =="
  if python -m lpse.apps.benchmark_llm_forecast \
    "${COMMON[@]}" \
    --llm-options "$mode" \
    --results-csv "${base}.csv" \
    --summary-json "${base}.json" \
    --llm-raw-out "${base}_raw.ndjson" \
    "$@"; then
    echo "OK $model/$cond/$mode"
  else
    echo "FAILED $model/$cond/$mode (continuing)"
  fi
}

run_cond qwen4b clean        open  --vllm-chat-url "$QWEN_URL" --vllm-model "$QWEN_MODEL"
run_cond qwen4b dropnode     open  --vllm-chat-url "$QWEN_URL" --vllm-model "$QWEN_MODEL" --exclude-node-ip "$EXCLUDE_NODE_IP"
run_cond qwen4b paraphrased  open  --vllm-chat-url "$QWEN_URL" --vllm-model "$QWEN_MODEL" --paraphrase-questions

run_cond nemotron clean       open  --vllm-chat-url "$NEMO_URL" --vllm-model "$NEMO_MODEL"
run_cond nemotron dropnode    open  --vllm-chat-url "$NEMO_URL" --vllm-model "$NEMO_MODEL" --exclude-node-ip "$EXCLUDE_NODE_IP"
run_cond nemotron paraphrased open  --vllm-chat-url "$NEMO_URL" --vllm-model "$NEMO_MODEL" --paraphrase-questions

run_cond qwen4b clean       category --vllm-chat-url "$QWEN_URL" --vllm-model "$QWEN_MODEL"
run_cond qwen4b paraphrased category --vllm-chat-url "$QWEN_URL" --vllm-model "$QWEN_MODEL" --paraphrase-questions
run_cond nemotron clean       category --vllm-chat-url "$NEMO_URL" --vllm-model "$NEMO_MODEL"
run_cond nemotron paraphrased category --vllm-chat-url "$NEMO_URL" --vllm-model "$NEMO_MODEL" --paraphrase-questions

echo "ALL RUNS COMPLETE"
