#!/usr/bin/env bash
# Task 1 (part A): Start the vLLM server configured for hidden-state extraction.
# Run in a SEPARATE terminal before 02b_generate_hidden_states.sh.
# Uses speculators_venv (needs vllm + speculators hidden-state connector).
set -euo pipefail

source speculators_venv/bin/activate

# model is positional; vllm passthrough args go after --
python speculators_repo/scripts/launch_vllm.py Qwen/Qwen3-8B \
    -- \
    --port 8000 \
    --dtype bfloat16

# This process must stay running while 02b runs.
