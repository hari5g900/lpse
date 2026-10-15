#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1
DUMP=artifacts/dumps/jepa_run
CKPT_BASE=artifacts/checkpoints/lpse_lap_eps15_10epck.pt
RES=artifacts/results/final
DONE_FILE=/tmp/epoch_eval.done

echo "[$(date)] epoch-eval watcher started"

while true; do
    if ! pgrep -f "train_lpse_offline" > /dev/null; then
        sleep 5
        echo "[$(date)] training process not running; final sweep"
        break
    fi
    for f in ${CKPT_BASE}.epoch*.pt; do
        [ -e "$f" ] || continue
        base=$(basename "$f" .pt)
        if [ -f "$RES/${base}_clean.json" ] && [ -f "$RES/${base}_para.json" ]; then
            continue
        fi
        s1=$(stat -c %s "$f" 2>/dev/null || echo 0)
        sleep 3
        s2=$(stat -c %s "$f" 2>/dev/null || echo 0)
        [ "$s1" = "$s2" ] || continue
        echo "[$(date)] evaluating $base"
        .venv/bin/python scripts/benchmark_offline_trace.py \
            --dump "$DUMP" --checkpoint "$f" \
            --snapshots 160 --questions-per-snapshot 5 --max-questions 800 --tail-fraction 0.2 --skip-llm \
            --results-csv "$RES/${base}_clean.csv" --summary-json "$RES/${base}_clean.json" 2>&1 | grep -E "^LPSE"
        .venv/bin/python scripts/benchmark_offline_trace.py \
            --dump "$DUMP" --checkpoint "$f" \
            --snapshots 160 --questions-per-snapshot 5 --max-questions 800 --tail-fraction 0.2 --skip-llm \
            --paraphrase-questions \
            --results-csv "$RES/${base}_para.csv" --summary-json "$RES/${base}_para.json" 2>&1 | grep -E "^LPSE"
        echo "[$(date)] done $base"
    done
    sleep 20
done

echo "[$(date)] watcher exiting; training finished"
echo done > "$DONE_FILE"
