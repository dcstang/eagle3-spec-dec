#!/usr/bin/env bash
# Full pipeline: runs all tasks in order, retries each step once on failure.
# Usage: bash run_all.sh
# Logs: logs/run_all_<timestamp>.log  (tee'd to stdout as well)
set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_all_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "  eagle3-spec-dec full pipeline  |  $(date)"
echo "  Log: $LOG_FILE"
echo "================================================================"

# ── helpers ──────────────────────────────────────────────────────────────────

VLLM_PID=""

run_step() {
    local name="$1"
    shift
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "STEP: $name"
    echo "CMD:  $*"
    echo "START: $(date)"
    echo "────────────────────────────────────────────────────────────"

    "$@"
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "WARN: '$name' failed (exit $rc). Retrying once …"
        "$@"
        rc=$?
    fi

    if [[ $rc -ne 0 ]]; then
        echo "ERROR: '$name' failed after retry. Aborting pipeline."
        kill_vllm
        exit $rc
    fi

    echo "OK: $name  ($(date))"
}

kill_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM (PID=$VLLM_PID) …"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    pkill -f "vllm serve" 2>/dev/null || true
    # Wait for the port to be fully released before starting the next server
    local port="${1:-8001}"
    local deadline=$(( SECONDS + 30 ))
    while lsof -i :"$port" -sTCP:LISTEN &>/dev/null; do
        if (( SECONDS > deadline )); then
            echo "WARN: port $port still in use after 30s — continuing anyway."
            break
        fi
        sleep 2
    done
}

wait_for_vllm() {
    local port="$1"
    # 600s: covers cold HuggingFace download (~16 GB) + model load on first run
    local deadline=$(( SECONDS + 600 ))
    echo "Waiting for vLLM on port $port (up to 600s) …"
    until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        if (( SECONDS > deadline )); then
            echo "ERROR: vLLM did not become healthy within 600s."
            kill_vllm "$port"
            return 1
        fi
        sleep 5
    done
    echo "vLLM ready."
}

# Run a single benchmark; --disable-prefix-caching is a server flag only,
# so it is NOT passed to vllm bench serve here.
run_bench() {
    local tag="$1"
    local model="$2"
    local port="$3"
    local out="results/${TIMESTAMP}_${tag}.txt"
    mkdir -p results

    echo "Benchmarking: $tag → $out"
    source vllm_venv/bin/activate
    vllm bench serve \
        --base-url "http://localhost:$port" \
        --model "$model" \
        --dataset-name hf \
        --max-concurrency 8 \
        --dataset-path philschmid/mt-bench \
        --num-prompts 80 \
        --seed 42 \
        2>&1 | tee "$out"
    local rc=${PIPESTATUS[0]}
    deactivate
    return $rc
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "Checking prerequisites …"
command -v python3.12 >/dev/null || { echo "ERROR: python3.12 not found."; exit 1; }
command -v git        >/dev/null || { echo "ERROR: git not found."; exit 1; }
command -v curl       >/dev/null || { echo "ERROR: curl not found."; exit 1; }
command -v lsof       >/dev/null || { echo "ERROR: lsof not found."; exit 1; }
echo "Prerequisites OK."

# ── Step 0: environment setup ─────────────────────────────────────────────────
run_step "00_setup_envs" bash scripts/00_setup_envs.sh

# ── Step 1: data preparation ──────────────────────────────────────────────────
run_step "01_prepare_data" bash scripts/01_prepare_data.sh

# ── Step 2: hidden-state generation ──────────────────────────────────────────
# Needs a vLLM server running from speculators_venv while generation runs.
echo ""
echo "────────────────────────────────────────────────────────────"
echo "STEP: 02_generate_hidden_states"
echo "START: $(date)"
echo "────────────────────────────────────────────────────────────"

launch_speculators_vllm() {
    kill_vllm 8000
    source speculators_venv/bin/activate
    python speculators_repo/scripts/launch_vllm.py \
        --model Qwen/Qwen3-8B \
        --port 8000 \
        --dtype bfloat16 \
        &
    VLLM_PID=$!
    deactivate
    wait_for_vllm 8000
}

generate_hidden_states() {
    rm -rf /tmp/hidden_states/*
    source speculators_venv/bin/activate
    python speculators_repo/scripts/generate_hidden_states.py \
        --model Qwen/Qwen3-8B \
        --data-dir data/sharegpt_processed \
        --output-dir data/hidden_states \
        --vllm-url http://localhost:8000 \
        --max-seq-len 2048
    local rc=$?
    deactivate
    return $rc
}

launch_speculators_vllm
gen_rc=0
generate_hidden_states || gen_rc=$?
kill_vllm 8000

if [[ $gen_rc -ne 0 ]]; then
    echo "WARN: hidden-state generation failed. Retrying once …"
    launch_speculators_vllm
    gen_rc=0
    generate_hidden_states || gen_rc=$?
    kill_vllm 8000
    if [[ $gen_rc -ne 0 ]]; then
        echo "ERROR: hidden-state generation failed after retry. Aborting."
        exit $gen_rc
    fi
fi
echo "OK: 02_generate_hidden_states  ($(date))"

# ── Step 3: EAGLE-3 training ──────────────────────────────────────────────────
run_step "03_train_eagle3" bash scripts/03_train_eagle3.sh

# ── Step 4: FP8 quantization ──────────────────────────────────────────────────
run_step "04_quantize_fp8" bash -c "
    source comp_venv/bin/activate
    python scripts/04_quantize_fp8.py
    deactivate
"

# ── Step 5: benchmarks ────────────────────────────────────────────────────────
BASE_MODEL="Qwen/Qwen3-8B"
FP8_MODEL="Qwen3-8B-FP8-Dynamic"
DRAFT_HEAD="output/checkpoints"
PORT=8001

# Pre-download the base model weights so the 600s timeout isn't eaten by download
echo "Pre-downloading $BASE_MODEL weights into HF cache …"
source vllm_venv/bin/activate
python -c "
from huggingface_hub import snapshot_download
snapshot_download('$BASE_MODEL')
" || true   # non-fatal: vllm serve will retry the download itself
deactivate

# 5a. Baseline
run_step "05a_bench_baseline" bash -c "
    source vllm_venv/bin/activate
    kill_vllm_local() {
        pkill -f 'vllm serve' 2>/dev/null || true
        local deadline=\$(( SECONDS + 30 ))
        while lsof -i :$PORT -sTCP:LISTEN &>/dev/null; do
            (( SECONDS > deadline )) && break; sleep 2
        done
    }
    kill_vllm_local
    vllm serve '$BASE_MODEL' \
        --port $PORT \
        --dtype bfloat16 \
        --disable-prefix-caching &
    SRV=\$!
    deadline=\$(( SECONDS + 600 ))
    until curl -sf http://localhost:$PORT/health &>/dev/null; do
        (( SECONDS > deadline )) && { echo 'ERROR: server timeout'; kill \$SRV; exit 1; }
        sleep 5
    done
    vllm bench serve \
        --base-url http://localhost:$PORT \
        --model '$BASE_MODEL' \
        --dataset-name hf --max-concurrency 8 \
        --dataset-path philschmid/mt-bench \
        --num-prompts 80 --seed 42 \
        2>&1 | tee results/${TIMESTAMP}_01_baseline.txt
    rc=\${PIPESTATUS[0]}
    kill \$SRV 2>/dev/null; wait \$SRV 2>/dev/null
    kill_vllm_local
    deactivate
    exit \$rc
"

# 5b. Speculative decoding (draft_tokens=2)
run_step "05b_bench_spec_dec" bash -c "
    source vllm_venv/bin/activate
    kill_vllm_local() {
        pkill -f 'vllm serve' 2>/dev/null || true
        local deadline=\$(( SECONDS + 30 ))
        while lsof -i :$PORT -sTCP:LISTEN &>/dev/null; do
            (( SECONDS > deadline )) && break; sleep 2
        done
    }
    kill_vllm_local
    vllm serve '$BASE_MODEL' \
        --port $PORT \
        --dtype bfloat16 \
        --disable-prefix-caching \
        --speculative-model '$DRAFT_HEAD' \
        --num-speculative-tokens 2 &
    SRV=\$!
    deadline=\$(( SECONDS + 600 ))
    until curl -sf http://localhost:$PORT/health &>/dev/null; do
        (( SECONDS > deadline )) && { echo 'ERROR: server timeout'; kill \$SRV; exit 1; }
        sleep 5
    done
    vllm bench serve \
        --base-url http://localhost:$PORT \
        --model '$BASE_MODEL' \
        --dataset-name hf --max-concurrency 8 \
        --dataset-path philschmid/mt-bench \
        --num-prompts 80 --seed 42 \
        2>&1 | tee results/${TIMESTAMP}_02_spec_dec_drafttokens2.txt
    rc=\${PIPESTATUS[0]}
    kill \$SRV 2>/dev/null; wait \$SRV 2>/dev/null
    kill_vllm_local
    deactivate
    exit \$rc
"

# 5c. FP8 only
run_step "05c_bench_fp8" bash -c "
    source vllm_venv/bin/activate
    kill_vllm_local() {
        pkill -f 'vllm serve' 2>/dev/null || true
        local deadline=\$(( SECONDS + 30 ))
        while lsof -i :$PORT -sTCP:LISTEN &>/dev/null; do
            (( SECONDS > deadline )) && break; sleep 2
        done
    }
    kill_vllm_local
    vllm serve '$FP8_MODEL' \
        --port $PORT \
        --dtype auto \
        --disable-prefix-caching &
    SRV=\$!
    deadline=\$(( SECONDS + 600 ))
    until curl -sf http://localhost:$PORT/health &>/dev/null; do
        (( SECONDS > deadline )) && { echo 'ERROR: server timeout'; kill \$SRV; exit 1; }
        sleep 5
    done
    vllm bench serve \
        --base-url http://localhost:$PORT \
        --model '$FP8_MODEL' \
        --dataset-name hf --max-concurrency 8 \
        --dataset-path philschmid/mt-bench \
        --num-prompts 80 --seed 42 \
        2>&1 | tee results/${TIMESTAMP}_03_fp8_only.txt
    rc=\${PIPESTATUS[0]}
    kill \$SRV 2>/dev/null; wait \$SRV 2>/dev/null
    kill_vllm_local
    deactivate
    exit \$rc
"

# 5d. FP8 + speculative decoding (draft_tokens=1)
run_step "05d_bench_fp8_spec" bash -c "
    source vllm_venv/bin/activate
    kill_vllm_local() {
        pkill -f 'vllm serve' 2>/dev/null || true
        local deadline=\$(( SECONDS + 30 ))
        while lsof -i :$PORT -sTCP:LISTEN &>/dev/null; do
            (( SECONDS > deadline )) && break; sleep 2
        done
    }
    kill_vllm_local
    vllm serve '$FP8_MODEL' \
        --port $PORT \
        --dtype auto \
        --disable-prefix-caching \
        --speculative-model '$DRAFT_HEAD' \
        --num-speculative-tokens 1 &
    SRV=\$!
    deadline=\$(( SECONDS + 600 ))
    until curl -sf http://localhost:$PORT/health &>/dev/null; do
        (( SECONDS > deadline )) && { echo 'ERROR: server timeout'; kill \$SRV; exit 1; }
        sleep 5
    done
    vllm bench serve \
        --base-url http://localhost:$PORT \
        --model '$FP8_MODEL' \
        --dataset-name hf --max-concurrency 8 \
        --dataset-path philschmid/mt-bench \
        --num-prompts 80 --seed 42 \
        2>&1 | tee results/${TIMESTAMP}_04_fp8_spec_dec_drafttokens1.txt
    rc=\${PIPESTATUS[0]}
    kill \$SRV 2>/dev/null; wait \$SRV 2>/dev/null
    kill_vllm_local
    deactivate
    exit \$rc
"

# ── Step 6: draft token sweep (FP8 + spec) ───────────────────────────────────
run_step "06_tune_draft_tokens" bash scripts/06_tune_draft_tokens.sh "$FP8_MODEL" "$DRAFT_HEAD"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  PIPELINE COMPLETE  |  $(date)"
echo "================================================================"
echo ""
echo "Output token throughput summary:"
grep "Output token throughput" results/${TIMESTAMP}_*.txt 2>/dev/null | \
    sed "s|results/${TIMESTAMP}_||;s|\.txt:||" || echo "(no result files found)"

echo ""
echo "Full log: $LOG_FILE"
echo "Benchmark files: results/${TIMESTAMP}_*.txt"
echo ""
echo "Paste the sections above into OUTPUTS.md."
