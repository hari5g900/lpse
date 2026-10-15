#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

LOGF="artifacts/logs/lap_autolaunch.log"
exec > >(tee -a "$LOGF") 2>&1

echo "[$(date)] Waiting for round 4 runner to finish..."
while pgrep -f "run_para_exp\.sh 4" > /dev/null; do
    sleep 30
done
echo "[$(date)] Round 4 done. Waiting for any residual training to release GPU..."
while pgrep -f train_lpse_offline > /dev/null; do
    sleep 15
done
sleep 20

echo "[$(date)] Smoke-testing LAP (should train without OOM)..."
rm -f /tmp/lap_smoke.csv /tmp/lap_smoke.pt
timeout 600 env LOAD_REPLAY_PATH=artifacts/dumps/jepa_run Q_ENCODER_BACKEND=hybrid STRUCTURED_QUESTION=0 \
    LAP_EPS=0.15 LAP_INNER_STEPS=2 \
    LPSE_RUN_SUFFIX=lap_smoke CHECKPOINT_PATH=/tmp/lap_smoke.pt LOG_CSV_PATH=/tmp/lap_smoke.csv \
    SUMMARY_INTERVAL=0 \
    .venv/bin/python train_lpse_offline.py > /tmp/lap_smoke_train.log 2>&1
RC=$?
if [ $RC -ne 0 ] && [ $RC -ne 124 ]; then
    echo "[$(date)] LAP SMOKE TEST FAILED (rc=$RC). ABORTING round 5."
    tail -15 /tmp/lap_smoke_train.log
    exit 1
fi
echo "[$(date)] LAP smoke test OK (rc=$RC). Checking it produced sane loss..."
if [ -f /tmp/lap_smoke.csv ]; then
    tail -1 /tmp/lap_smoke.csv
fi

echo "[$(date)] Launching round 5 (LAP variants)..."
bash scripts/run_para_exp.sh 5 > artifacts/logs/para_exp5_runner.log 2>&1
echo "[$(date)] Round 5 complete."
