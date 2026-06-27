#!/usr/bin/env bash
# Task 2: Train the EAGLE-3 draft head using precomputed hidden states.
# Uses speculators_venv.
# Checkpoints → output/checkpoints/
set -uo pipefail   # no -e: run_all.sh owns retry/abort logic

source speculators_venv/bin/activate

MODEL="Qwen/Qwen3-8B"
DATA_PATH="data/sharegpt_processed"          # root dir with preprocessed data + token_freq.pt
HIDDEN_STATES_PATH="data/hidden_states"      # precomputed hidden states
SAVE_PATH="output/checkpoints"

mkdir -p "$SAVE_PATH"

python speculators_repo/scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --speculator-type eagle3 \
    --draft-arch qwen3 \
    --data-path "$DATA_PATH" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --on-missing raise \
    --save-path "$SAVE_PATH" \
    --epochs 5 \
    --lr 1e-3 \
    --scheduler-warmup-steps 100 \
    --seed 42 \
    --log-freq 50 \
    --save-best \
    --checkpoint-freq 1

echo "Training complete. Checkpoints → $SAVE_PATH"
echo "Use the checkpoint marked best (lowest val/loss_epoch) for serving."

deactivate
