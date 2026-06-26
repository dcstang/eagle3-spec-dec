#!/usr/bin/env bash
# Task 2: Train the EAGLE-3 draft head using precomputed hidden states.
# Uses speculators_venv.
# Checkpoints → output/checkpoints/
set -uo pipefail   # no -e: run_all.sh owns retry/abort logic

source speculators_venv/bin/activate

MODEL="Qwen/Qwen3-8B"
HIDDEN_STATES_DIR="data/hidden_states"
OUTPUT_DIR="output/checkpoints"

mkdir -p "$OUTPUT_DIR"

python speculators_repo/scripts/train_eagle3.py \
    --model "$MODEL" \
    --hidden-states-dir "$HIDDEN_STATES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 5 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --lr 1e-3 \
    --warmup-steps 100 \
    --seed 42 \
    --log-every 50

echo "Training complete. Checkpoints → $OUTPUT_DIR"
echo "Pick the checkpoint with the lowest val/loss_epoch for serving."

deactivate
