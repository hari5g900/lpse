#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

ROUND="${1:-}"
case "$ROUND" in
    2|3|4|5|6) ;;
    *) echo "usage: $0 <round 2-6>" >&2; exit 1 ;;
esac

DUMP="${DUMP:-artifacts/dumps/jepa_run}"
CKPT_DIR="artifacts/checkpoints/para_exp${ROUND}"
RESULT_DIR="artifacts/results/para_exp${ROUND}"
LOG_DIR="artifacts/logs/para_exp${ROUND}"
mkdir -p "$CKPT_DIR" "$RESULT_DIR" "$LOG_DIR"

BASE_ENV=(LOAD_REPLAY_PATH="$DUMP" Q_ENCODER_BACKEND=hybrid STRUCTURED_QUESTION=0)

run_variant () {
    local name="$1"; shift
    local envs=("${BASE_ENV[@]}" "$@")
    local ckpt="$CKPT_DIR/lpse_${name}.pt"
    local csv="$LOG_DIR/lpse_${name}.csv"
    echo "===== TRAIN $name ====="
    env "${envs[@]}" LPSE_RUN_SUFFIX="para${ROUND}_${name}" \
        CHECKPOINT_PATH="$ckpt" LOG_CSV_PATH="$csv" \
        SUMMARY_INTERVAL=0 \
        .venv/bin/python -m lpse.apps.train_lpse_offline > "$LOG_DIR/${name}_train.log" 2>&1
    tail -1 "$LOG_DIR/${name}_train.log"
    echo "===== BENCH $name (clean) ====="
    env "${envs[@]}" .venv/bin/python scripts/eval/benchmark_offline_trace.py \
        --dump "$DUMP" --checkpoint "$ckpt" \
        --snapshots 40 --questions-per-snapshot 5 --max-questions 100 --tail-fraction 0.2 --skip-llm \
        --results-csv "$RESULT_DIR/${name}_clean.csv" --summary-json "$RESULT_DIR/${name}_clean.json" 2>&1 | grep -E "^LPSE"
    echo "===== BENCH $name (paraphrased) ====="
    env "${envs[@]}" .venv/bin/python scripts/eval/benchmark_offline_trace.py \
        --dump "$DUMP" --checkpoint "$ckpt" \
        --snapshots 40 --questions-per-snapshot 5 --max-questions 100 --tail-fraction 0.2 --skip-llm \
        --paraphrase-questions \
        --results-csv "$RESULT_DIR/${name}_para.csv" --summary-json "$RESULT_DIR/${name}_para.json" 2>&1 | grep -E "^LPSE"
}

case "$ROUND" in
2)
    echo "## A-light (noise=0.05, dropout=0.1)"
    run_variant A_light Q_EMB_NOISE_STD=0.05 Q_EMB_DROPOUT=0.1

    echo "## A-strong (noise=0.15, dropout=0.3)"
    run_variant A_strong Q_EMB_NOISE_STD=0.15 Q_EMB_DROPOUT=0.3

    echo "## A+B (noise=0.1, dropout=0.2, tau=0.12, dropout=0.2)"
    run_variant AplusB Q_EMB_NOISE_STD=0.1 Q_EMB_DROPOUT=0.2 CODEBOOK_TAU=0.12 DROPOUT=0.2

    echo "## T-text-only-noise (TEXT_NOISE_STD=0.15, TEXT_DROPOUT=0.2)"
    run_variant T_text TEXT_NOISE_STD=0.15 TEXT_DROPOUT=0.2

    echo "## T-strong (TEXT_NOISE_STD=0.25, TEXT_DROPOUT=0.3)"
    run_variant T_strong TEXT_NOISE_STD=0.25 TEXT_DROPOUT=0.3

    echo "## T-text + B (TEXT noise=0.15/drop=0.2, tau=0.12, dropout=0.2)"
    run_variant TplusB TEXT_NOISE_STD=0.15 TEXT_DROPOUT=0.2 CODEBOOK_TAU=0.12 DROPOUT=0.2
    ;;
3)
    echo "## Ak3 (A + JEPA k=3)"
    run_variant Ak3 Q_EMB_NOISE_STD=0.1 Q_EMB_DROPOUT=0.2 JEPA_PRED_HORIZON=3

    echo "## Ak5 (A + JEPA k=5)"
    run_variant Ak5 Q_EMB_NOISE_STD=0.1 Q_EMB_DROPOUT=0.2 JEPA_PRED_HORIZON=5

    echo "## Ak7 (A + JEPA k=7)"
    run_variant Ak7 Q_EMB_NOISE_STD=0.1 Q_EMB_DROPOUT=0.2 JEPA_PRED_HORIZON=7
    ;;
4)
    echo "## ADV-fgsm (eps=0.15, steps=1)"
    run_variant ADV_fgsm ADV_EPS=0.15 ADV_STEPS=1

    echo "## ADV-pgd3 (eps=0.15, steps=3)"
    run_variant ADV_pgd3 ADV_EPS=0.15 ADV_STEPS=3

    echo "## ADV-pgd5 (eps=0.15, steps=5)"
    run_variant ADV_pgd5 ADV_EPS=0.15 ADV_STEPS=5

    echo "## ADV-eps30 (eps=0.30, steps=3)"
    run_variant ADV_eps30 ADV_EPS=0.30 ADV_STEPS=3

    echo "## A+ADV (noise 0.1/0.2 + eps=0.10 steps=2)"
    run_variant AplusADV Q_EMB_NOISE_STD=0.1 Q_EMB_DROPOUT=0.2 ADV_EPS=0.10 ADV_STEPS=2
    ;;
5)
    echo "## LAP-eps10 (eps=0.10, steps=2)"
    run_variant LAP_eps10 LAP_EPS=0.10 LAP_INNER_STEPS=2

    echo "## LAP-eps15 (eps=0.15, steps=2)"
    run_variant LAP_eps15 LAP_EPS=0.15 LAP_INNER_STEPS=2

    echo "## LAP-eps15s3 (eps=0.15, steps=3)"
    run_variant LAP_eps15s3 LAP_EPS=0.15 LAP_INNER_STEPS=3

    echo "## LAP+A (eps=0.10 + lever A noise 0.1/0.2)"
    run_variant LAPplusA LAP_EPS=0.10 LAP_INNER_STEPS=2 Q_EMB_NOISE_STD=0.1 Q_EMB_DROPOUT=0.2

    echo "## LAP-eps20 (eps=0.20, steps=2)"
    run_variant LAP_eps20 LAP_EPS=0.20 LAP_INNER_STEPS=2
    ;;
6)
    echo "## eps=0.12"
    run_variant eps12 LAP_EPS=0.12 LAP_INNER_STEPS=2
    echo "## eps=0.13"
    run_variant eps13 LAP_EPS=0.13 LAP_INNER_STEPS=2
    echo "## eps=0.14"
    run_variant eps14 LAP_EPS=0.14 LAP_INNER_STEPS=2
    echo "## eps=0.16"
    run_variant eps16 LAP_EPS=0.16 LAP_INNER_STEPS=2
    echo "## eps=0.17"
    run_variant eps17 LAP_EPS=0.17 LAP_INNER_STEPS=2
    echo "## eps=0.18"
    run_variant eps18 LAP_EPS=0.18 LAP_INNER_STEPS=2
    ;;
esac

echo "=== ALL DONE ==="
