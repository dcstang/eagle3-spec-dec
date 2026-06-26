#!/usr/bin/env bash
# Task 1 (part A): Start the vLLM server that the hidden-state generator calls.
# Run this in a SEPARATE terminal / tmux pane before 02b_generate_hidden_states.sh.
# Uses speculators_venv (speculators bundles its own serving utilities).
set -euo pipefail

source speculators_venv/bin/activate

MODEL="Qwen/Qwen3-8B"
PORT=8000

python speculators_repo/scripts/launch_vllm.py \
    --model "$MODEL" \
    --port "$PORT" \
    --dtype bfloat16

# This process must stay running while 02b runs.
