#!/usr/bin/env bash
# Task 1: Download and tokenise ShareGPT data into the format speculators expects.
# Uses speculators_venv.
set -uo pipefail   # no -e: run_all.sh owns retry/abort logic

source speculators_venv/bin/activate

MODEL="Qwen/Qwen3-8B"
MAX_SAMPLES=3000
SEQ_LEN=2048
OUTPUT_DIR="data/sharegpt_processed"

mkdir -p "$OUTPUT_DIR"

python speculators_repo/scripts/prepare_data.py \
    --model "$MODEL" \
    --data sharegpt \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LEN" \
    --output "$OUTPUT_DIR"

echo "Data prepared → $OUTPUT_DIR"
deactivate
