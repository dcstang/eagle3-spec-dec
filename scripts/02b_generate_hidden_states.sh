#!/usr/bin/env bash
# Task 1 (part B): Generate offline hidden states for EAGLE-3 training.
# Requires 02a_launch_vllm_for_hidden_states.sh to be running first.
# ~140 GB disk space for 3000 samples — monitor with `df -h`.
# Uses speculators_venv.
set -uo pipefail   # no -e: run_all.sh owns retry/abort logic

source speculators_venv/bin/activate

MODEL="Qwen/Qwen3-8B"
PREPROCESSED_DIR="data/sharegpt_processed"
HIDDEN_STATES_DIR="data/hidden_states"

# Clear stale temp files if re-running after a failure
rm -rf /tmp/hidden_states/*

mkdir -p "$HIDDEN_STATES_DIR"

python speculators_repo/scripts/data_generation_offline.py \
    --model "$MODEL" \
    --preprocessed-data "$PREPROCESSED_DIR" \
    --output "$HIDDEN_STATES_DIR" \
    --endpoint http://localhost:8000/v1 \
    --concurrency 8 \
    --validate-outputs

echo "Hidden states saved → $HIDDEN_STATES_DIR"
echo "Disk usage:"
du -sh "$HIDDEN_STATES_DIR"

deactivate
