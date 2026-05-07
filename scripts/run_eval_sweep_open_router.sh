#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_eval_sweep_open_router.sh
#
# Runs ALL benchmark evaluations over a sweep of OpenRouter models.
# Each model is evaluated with:
#   python -m scripts.chat_eval_open_router -m <model> -a ALL \
#       --gpqa-mode 0-shot-cot few-shot-cot --log
#
# Failures exit immediately (non-recoverable for API-based evals).
#
# Logs: logs/eval_sweep_open_router/<model-slug>.log  (stdout + stderr per model)
# Results: written by chat_eval_open_router.py to .cache/nanollama/results/
#
# Usage:
#   bash scripts/run_eval_sweep_open_router.sh
#   OPENROUTER_API_KEY=sk-... bash scripts/run_eval_sweep_open_router.sh
# ---------------------------------------------------------------------------

set -uo pipefail

MODELS=(
    "meta-llama/llama-3.2-1b-instruct"
    "meta-llama/llama-3.2-3b-instruct"
    "meta-llama/llama-3.1-8b-instruct"
    "meta-llama/llama-3.1-70b-instruct"
    "nousresearch/hermes-3-llama-3.1-405b"
)

LOG_DIR="logs/eval_sweep_open_router"
mkdir -p "$LOG_DIR"

total=${#MODELS[@]}
idx=0

for model in "${MODELS[@]}"; do
    idx=$((idx + 1))
    slug=$(echo "$model" | tr '/' '-')
    log_file="${LOG_DIR}/${slug}.log"

    echo ""
    echo "=========================================="
    echo "[$idx/$total] $model"
    echo "  log  -> $log_file"
    echo "=========================================="

    python -m scripts.chat_eval_open_router \
        -m "$model" \
        -a ALL \
        --gpqa-mode 0-shot-cot few-shot-cot \
        --log \
        2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -eq 0 ]; then
        echo "  DONE -- $model"
    else
        echo ""
        echo "  ERROR: $model failed (exit_code=$exit_code)."
        echo "  See $log_file for details. Exiting sweep."
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Sweep complete. $total/$total models evaluated."
echo "Results in: .cache/nanollama/results/"
echo "Logs in:    $LOG_DIR/"
echo "=========================================="
