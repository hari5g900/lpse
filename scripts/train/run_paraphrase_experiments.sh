#!/usr/bin/env bash
set -euo pipefail

DUMP="${DUMP:-artifacts/dumps/jepa_run}"
STEPS="${STEPS:-30000}"
CKPT_DIR="${CKPT_DIR:-artifacts/checkpoints/para_exp}"
RESULT_DIR="${RESULT_DIR:-artifacts/results/para_exp}"
LOG_DIR="${LOG_DIR:-artifacts/logs/para_exp}"
mkdir -p "$CKPT_DIR" "$RESULT_DIR" "$LOG_DIR"

BASE_ENV=(LOAD_REPLAY_PATH="$DUMP" Q_ENCODER_BACKEND=hybrid STRUCTURED_QUESTION=0)

run_variant () {
    local name="$1"; shift
    local envs=("${BASE_ENV[@]}" "$@")
    local ckpt="$CKPT_DIR/lpse_${name}.pt"
    local csv="$LOG_DIR/lpse_${name}.csv"
    echo "===== TRAIN $name ====="
    env "${envs[@]}" LPSE_RUN_SUFFIX="para_${name}" \
        CHECKPOINT_PATH="$ckpt" LOG_CSV_PATH="$csv" \
        SUMMARY_INTERVAL=0 \
        .venv/bin/python -m lpse.apps.train_lpse_offline 2>&1 | tail -5
    echo "===== BENCH $name (clean) ====="
    env "${envs[@]}" .venv/bin/python scripts/eval/benchmark_offline_trace.py \
        --dump "$DUMP" --checkpoint "$ckpt" \
        --snapshots 40 --questions-per-snapshot 5 --max-questions 100 --tail-fraction 0.2 --skip-llm \
        --results-csv "$RESULT_DIR/${name}_clean.csv" --summary-json "$RESULT_DIR/${name}_clean.json" 2>&1 | grep -E "LPSE|MLP|XGB"
    echo "===== BENCH $name (paraphrased) ====="
    env "${envs[@]}" .venv/bin/python scripts/eval/benchmark_offline_trace.py \
        --dump "$DUMP" --checkpoint "$ckpt" \
        --snapshots 40 --questions-per-snapshot 5 --max-questions 100 --tail-fraction 0.2 --skip-llm \
        --paraphrase-questions \
        --results-csv "$RESULT_DIR/${name}_para.csv" --summary-json "$RESULT_DIR/${name}_para.json" 2>&1 | grep -E "LPSE|MLP|XGB"
}

echo "## Variant: baseline (no levers)"
run_variant baseline

echo "## Variant: A only (text-path perturbation: noise=0.1, dropout=0.2)"
run_variant A Q_EMB_NOISE_STD=0.1 Q_EMB_DROPOUT=0.2

echo "## Variant: B only (softer readout: tau=0.12, dropout=0.2)"
run_variant B CODEBOOK_TAU=0.12 DROPOUT=0.2

echo "## Variant: C only (JEPA horizon k=3)"
run_variant C JEPA_PRED_HORIZON=3

echo "## Variant: D only (token-sequence access)"
run_variant D TOKEN_QUESTION=1

echo "## Variant: ALL (A+B+C+D)"
run_variant ALL Q_EMB_NOISE_STD=0.1 Q_EMB_DROPOUT=0.2 CODEBOOK_TAU=0.12 DROPOUT=0.2 JEPA_PRED_HORIZON=3 TOKEN_QUESTION=1

echo "=== ALL DONE ==="
