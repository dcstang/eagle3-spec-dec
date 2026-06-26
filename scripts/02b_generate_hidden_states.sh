#!/usr/bin/env bash
# Task 1 (part B): Generate offline hidden states for EAGLE-3 training.
# Requires 02a_launch_vllm_for_hidden_states.sh to be running first.
# ~140 GB disk space for 3 000 samples at seq_len=2048 — monitor with `df -h`.
# Uses speculators_venv.
set -euo pipefail

source speculators_venv/bin/activate

MODEL="Qwen/Qwen3-8B"
DATA_DIR="data/sharegpt_processed"
HIDDEN_STATES_DIR="data/hidden_states"
PORT=8000

# Clear stale temp files if re-running after a failure
rm -rf /tmp/hidden_states/*

mkdir -p "$HIDDEN_STATES_DIR"

python speculators_repo/scripts/generate_hidden_states.py \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$HIDDEN_STATES_DIR" \
    --vllm-url "http://localhost:$PORT" \
    --max-seq-len 2048

echo "Hidden states saved → $HIDDEN_STATES_DIR"
echo "Disk usage:"
du -sh "$HIDDEN_STATES_DIR"

deactivate
