#!/usr/bin/env bash
# Sweep num_speculative_tokens (1, 2, 3) for a given model + draft head.
# Usage: bash scripts/06_tune_draft_tokens.sh <model> <draft_head_path>
# Uses vllm_venv.
set -uo pipefail

source vllm_venv/bin/activate

MODEL="${1:-Qwen3-8B-FP8-Dynamic}"
DRAFT_HEAD="${2:-output/checkpoints}"
PORT=8001
RESULTS_DIR="results/tuning"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

wait_for_server() {
    local deadline=$(( SECONDS + 600 ))
    until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
        (( SECONDS > deadline )) && { echo "ERROR: server timeout"; return 1; }
        sleep 5
    done
}

kill_server() {
    local pid
    pid=$(lsof -ti :"$PORT" -sTCP:LISTEN 2>/dev/null || true)
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    local deadline=$(( SECONDS + 30 ))
    while lsof -i :"$PORT" -sTCP:LISTEN &>/dev/null; do
        (( SECONDS > deadline )) && break; sleep 2
    done
}

for N in 1 2 3; do
    echo "=== draft_tokens=$N ==="
    kill_server
    # vllm 0.20.0: speculative decoding configured via --speculative-config JSON
    vllm serve "$MODEL" \
        --port "$PORT" \
        --dtype auto \
        --disable-prefix-caching \
        --speculative-config "{\"model\": \"$DRAFT_HEAD\", \"num_speculative_tokens\": $N, \"method\": \"eagle3\"}" \
        --trust-remote-code \
        &
    SRV=$!
    wait_for_server

    OUT="$RESULTS_DIR/${TIMESTAMP}_drafttokens${N}.txt"
    vllm bench serve \
        --base-url "http://localhost:$PORT" \
        --model "$MODEL" \
        --dataset-name hf \
        --max-concurrency 8 \
        --dataset-path philschmid/mt-bench \
        --num-prompts 80 \
        --seed 42 \
        2>&1 | tee "$OUT"

    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    kill_server
done

echo "Sweep complete → $RESULTS_DIR/"
grep "Output token throughput" "$RESULTS_DIR/${TIMESTAMP}"_*.txt

deactivate
