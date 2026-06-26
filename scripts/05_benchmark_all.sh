#!/usr/bin/env bash
# Task 4: Benchmark all four configurations.
# Each config: start vLLM server → run vllm bench serve → kill server → log results.
# Uses vllm_venv.
# Outputs saved to results/ with timestamps.
set -euo pipefail

source vllm_venv/bin/activate

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

DRAFT_HEAD="output/checkpoints"   # path to best EAGLE-3 checkpoint
FP8_MODEL="Qwen3-8B-FP8-Dynamic"
BASE_MODEL="Qwen/Qwen3-8B"
PORT=8001
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

BENCH_ARGS=(
    --base-url "http://localhost:$PORT"
    --model "$BASE_MODEL"
    --dataset-name hf
    --max-concurrency 8
    --dataset-path philschmid/mt-bench
    --num-prompts 80
    --seed 42
    --disable-prefix-caching
)

wait_for_server() {
    local url="http://localhost:$PORT/health"
    echo "Waiting for vLLM at $url …"
    until curl -sf "$url" > /dev/null 2>&1; do sleep 2; done
    echo "Server ready."
}

kill_server() {
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
}

run_bench() {
    local tag="$1"
    local out="$RESULTS_DIR/${TIMESTAMP}_${tag}.txt"
    echo "=== Benchmarking: $tag ===" | tee -a "$out"
    vllm bench serve "${BENCH_ARGS[@]}" 2>&1 | tee -a "$out"
    echo "Results → $out"
}

# ── 1. Baseline ──────────────────────────────────────────────────────────────
kill_server
vllm serve "$BASE_MODEL" \
    --port "$PORT" \
    --dtype bfloat16 \
    --disable-prefix-caching \
    &
wait_for_server
run_bench "01_baseline"
kill_server

# ── 2. Speculative decoding (EAGLE-3, draft_tokens=2 as starting point) ──────
kill_server
vllm serve "$BASE_MODEL" \
    --port "$PORT" \
    --dtype bfloat16 \
    --disable-prefix-caching \
    --speculative-model "$DRAFT_HEAD" \
    --num-speculative-tokens 2 \
    &
wait_for_server
run_bench "02_spec_dec_drafttokens2"
kill_server

# ── 3. FP8 quantization (no speculative decoding) ────────────────────────────
kill_server
vllm serve "$FP8_MODEL" \
    --port "$PORT" \
    --dtype auto \
    --disable-prefix-caching \
    &
wait_for_server
BENCH_ARGS[1]="$FP8_MODEL"   # update --model to FP8 model
run_bench "03_fp8_only"
BENCH_ARGS[1]="$BASE_MODEL"  # restore
kill_server

# ── 4. FP8 + speculative decoding (draft_tokens=1 as starting point) ─────────
kill_server
vllm serve "$FP8_MODEL" \
    --port "$PORT" \
    --dtype auto \
    --disable-prefix-caching \
    --speculative-model "$DRAFT_HEAD" \
    --num-speculative-tokens 1 \
    &
wait_for_server
BENCH_ARGS[1]="$FP8_MODEL"
run_bench "04_fp8_spec_dec_drafttokens1"
BENCH_ARGS[1]="$BASE_MODEL"
kill_server

echo ""
echo "All benchmarks complete. Results in $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"

deactivate
