#!/usr/bin/env bash
# Optional: sweep num-speculative-tokens (1, 2, 3) for a given model config
# to find the optimal value. Pass MODEL and optional DRAFT_HEAD path.
# Uses vllm_venv.
set -euo pipefail

source vllm_venv/bin/activate

MODEL="${1:-Qwen3-8B-FP8-Dynamic}"
DRAFT_HEAD="${2:-output/checkpoints}"
PORT=8001
RESULTS_DIR="results/tuning"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

wait_for_server() {
    until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do sleep 2; done
}
kill_server() { pkill -f "vllm serve" 2>/dev/null || true; sleep 3; }

for N in 1 2 3; do
    echo "=== draft_tokens=$N ==="
    kill_server
    vllm serve "$MODEL" \
        --port "$PORT" \
        --dtype auto \
        --disable-prefix-caching \
        --speculative-model "$DRAFT_HEAD" \
        --num-speculative-tokens "$N" \
        &
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
        --disable-prefix-caching \
        2>&1 | tee "$OUT"

    kill_server
done

echo "Sweep complete → $RESULTS_DIR/"
grep "Output token throughput" "$RESULTS_DIR/${TIMESTAMP}"_*.txt

deactivate
