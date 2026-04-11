#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_eval_sweep_hf.sh
#
# Runs CORE benchmark evaluation over a sweep of HuggingFace models.
# Each model is evaluated with:
#   python -m scripts.base_eval --hf-path <model> --eval core --device-batch-size <bs>
#
# OOM retry strategy:
#   - First attempt:  --device-batch-size 4
#   - On OOM, retry: --device-batch-size 2
#   - On OOM, retry: --device-batch-size 1
#   - If still OOM at batch size 1, exit with error.
#   - Non-OOM failures also exit immediately (can't auto-recover).
#
# Logs: logs/eval_sweep/<model-slug>.log  (stdout + stderr per model)
# CSVs: written by base_eval.py to $NANOCHAT_BASE_DIR/base_eval/<model-slug>.csv
#        (NANOCHAT_BASE_DIR defaults to "out" if unset)
#
# Usage:
#   bash scripts/run_eval_sweep_hf.sh
#   HF_TOKEN=hf_... bash scripts/run_eval_sweep_hf.sh   # for gated models
# ---------------------------------------------------------------------------

set -uo pipefail

MODELS=(
    # --- OpenAI GPT-2 family ---
    "openai-community/gpt2"
    "openai-community/gpt2-medium"
    "openai-community/gpt2-large"
    "openai-community/gpt2-xl"
    # --- Meta LLama 2 chat ---
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Llama-2-70b-chat-hf"
    # --- Meta LLama 3 base ---
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3-70B"
    # --- Meta LLama 3.1 base ---
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.1-70B"
)

BATCH_SIZES=(4 2 1)
LOG_DIR="logs/eval_sweep"
mkdir -p "$LOG_DIR"

# Returns 0 if the log file contains an OOM error message.
is_oom() {
    grep -qi "out of memory\|OutOfMemoryError\|CUDA out of memory" "$1"
}

# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

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

    success=false

    for bs in "${BATCH_SIZES[@]}"; do
        echo "  [batch_size=$bs] running..."

        # Run eval; tee to log; capture python's exit code separately from tee's.
        # python -m scripts.base_eval \
        #     --hf-path "$model" \
        #     --eval core \
        #     --device-batch-size="$bs" \
        #     2>&1 | tee "$log_file"
        # exit_code=${PIPESTATUS[0]}

        torchrun --nproc_per_node=4 -m scripts.base_eval \
            --hf-path "$model" \
            --eval core \
            --device-batch-size="$bs" \
            2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}

        if [ "$exit_code" -eq 0 ]; then
            echo "  [batch_size=$bs] DONE -- $model"
            success=true
            break
        fi

        # Distinguish OOM from other failures.
        if is_oom "$log_file"; then
            if [ "$bs" -eq 1 ]; then
                echo ""
                echo "  ERROR: OOM even at batch_size=1 for $model."
                echo "  Cannot reduce batch size further. Exiting sweep."
                exit 1
            fi
            echo "  OOM detected at batch_size=$bs -- retrying with smaller batch..."
        else
            echo ""
            echo "  ERROR: $model failed with a non-OOM error (exit_code=$exit_code)."
            echo "  See $log_file for details. Exiting sweep."
            exit 1
        fi
    done

    if [ "$success" = false ]; then
        # Shouldn't reach here, but guard just in case.
        echo "  ERROR: all batch sizes exhausted for $model. Exiting."
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Sweep complete. $total/$total models evaluated."
echo "CSVs in: \${NANOCHAT_BASE_DIR:-out}/base_eval/"
echo "Logs in: $LOG_DIR/"
echo "=========================================="
