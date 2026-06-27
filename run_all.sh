#!/usr/bin/env bash
# Full pipeline: runs all tasks in order, retries each step once on failure.
# Usage:
#   bash run_all.sh                     # run everything from the start
#   START_FROM=05a bash run_all.sh      # skip to a specific step
#
# Step IDs: 00 01 02 03 04 05a 05b 05c 05d 06
#
# Logs: logs/run_all_<timestamp>.log  (tee'd to stdout as well)
set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_all_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

# Step to resume from (default: start from the beginning)
START_FROM="${START_FROM:-00}"

echo "================================================================"
echo "  eagle3-spec-dec full pipeline  |  $(date)"
echo "  Log: $LOG_FILE"
echo "  Resuming from step: $START_FROM"
echo "================================================================"

# ── helpers ──────────────────────────────────────────────────────────────────

VLLM_PID=""
SKIP=true   # flip to false once we reach START_FROM

should_run() {
    local id="$1"
    if $SKIP && [[ "$id" == "$START_FROM"* || "$id" > "$START_FROM" || "$id" == "$START_FROM" ]]; then
        SKIP=false
    fi
    ! $SKIP
}

run_step() {
    local id="$1"
    local name="$2"
    shift 2

    if ! should_run "$id"; then
        echo "SKIP: $name (before START_FROM=$START_FROM)"
        return 0
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "STEP [$id]: $name"
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
    local port="${1:-8001}"
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM (PID=$VLLM_PID) …"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    # Kill by exact port rather than name pattern to avoid matching this script
    local pid
    pid=$(lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null || true)
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    # Wait for port to free
    local deadline=$(( SECONDS + 30 ))
    while lsof -i :"$port" -sTCP:LISTEN &>/dev/null; do
        (( SECONDS > deadline )) && { echo "WARN: port $port still busy after 30s"; break; }
        sleep 2
    done
}

wait_for_vllm() {
    local port="$1"
    local deadline=$(( SECONDS + 600 ))
    echo "Waiting for vLLM on port $port (up to 600s) …"
    until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        if (( SECONDS > deadline )); then
            echo "ERROR: vLLM did not become healthy within 600s."
            return 1
        fi
        sleep 5
    done
    echo "vLLM ready."
}

# Run a benchmark; server must already be up. Kills server by PID when done.
run_bench() {
    local tag="$1"
    local model="$2"
    local port="$3"
    local srv_pid="$4"
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

    kill "$srv_pid" 2>/dev/null || true
    wait "$srv_pid" 2>/dev/null || true
    kill_vllm "$port"
    return $rc
}

# Start vllm serve; sets global SRV_PID
start_vllm_serve() {
    local port="$1"
    shift
    kill_vllm "$port"
    source vllm_venv/bin/activate
    vllm serve "$@" --port "$port" &
    SRV_PID=$!
    deactivate
    wait_for_vllm "$port"
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "Checking prerequisites …"
command -v python3.12 >/dev/null || { echo "ERROR: python3.12 not found."; exit 1; }
command -v git        >/dev/null || { echo "ERROR: git not found."; exit 1; }
command -v curl       >/dev/null || { echo "ERROR: curl not found."; exit 1; }
command -v lsof       >/dev/null || { echo "ERROR: lsof not found."; exit 1; }
echo "Prerequisites OK."

# ── Step 00: environment setup ────────────────────────────────────────────────
run_step "00" "setup_envs" bash scripts/00_setup_envs.sh

# ── Step 01: data preparation ─────────────────────────────────────────────────
run_step "01" "prepare_data" bash scripts/01_prepare_data.sh

# ── Step 02: hidden-state generation ─────────────────────────────────────────
if should_run "02"; then
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "STEP [02]: generate_hidden_states"
    echo "START: $(date)"
    echo "────────────────────────────────────────────────────────────"

    launch_speculators_vllm() {
        kill_vllm 8000
        source speculators_venv/bin/activate
        python speculators_repo/scripts/launch_vllm.py Qwen/Qwen3-8B \
            -- --port 8000 --dtype bfloat16 &
        VLLM_PID=$!
        deactivate
        wait_for_vllm 8000
    }

    generate_hidden_states() {
        rm -rf /tmp/hidden_states/*
        source speculators_venv/bin/activate
        python speculators_repo/scripts/data_generation_offline.py \
            --model Qwen/Qwen3-8B \
            --preprocessed-data data/sharegpt_processed \
            --output data/hidden_states \
            --endpoint http://localhost:8000/v1 \
            --concurrency 8 \
            --validate-outputs
        local rc=$?
        deactivate
        return $rc
    }

    launch_speculators_vllm
    gen_rc=0; generate_hidden_states || gen_rc=$?
    kill_vllm 8000

    if [[ $gen_rc -ne 0 ]]; then
        echo "WARN: hidden-state generation failed. Retrying once …"
        launch_speculators_vllm
        gen_rc=0; generate_hidden_states || gen_rc=$?
        kill_vllm 8000
        if [[ $gen_rc -ne 0 ]]; then
            echo "ERROR: hidden-state generation failed after retry. Aborting."
            exit $gen_rc
        fi
    fi
    echo "OK: generate_hidden_states  ($(date))"
fi

# ── Step 03: EAGLE-3 training ─────────────────────────────────────────────────
run_step "03" "train_eagle3" bash scripts/03_train_eagle3.sh

# ── Step 04: FP8 quantization ─────────────────────────────────────────────────
run_step "04" "quantize_fp8" bash -c "
    source comp_venv/bin/activate
    python scripts/04_quantize_fp8.py
    deactivate
"

# ── Step 05: benchmarks ───────────────────────────────────────────────────────
BASE_MODEL="Qwen/Qwen3-8B"
FP8_MODEL="Qwen3-8B-FP8-Dynamic"
DRAFT_HEAD="output/checkpoints"
PORT=8001
SRV_PID=""

# 5a. Baseline
run_step "05a" "bench_baseline" bash -c "
    source vllm_venv/bin/activate
    vllm serve '$BASE_MODEL' \
        --port $PORT \
        --dtype bfloat16 \
        &
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
    # kill by port — avoids pkill pattern matching this script's cmdline
    pid=\$(lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null || true)
    [[ -n \"\$pid\" ]] && kill \"\$pid\" 2>/dev/null || true
    deactivate
    exit \$rc
"

# 5b. Speculative decoding (draft_tokens=2)
run_step "05b" "bench_spec_dec" bash -c "
    source vllm_venv/bin/activate
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
    pid=\$(lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null || true)
    [[ -n \"\$pid\" ]] && kill \"\$pid\" 2>/dev/null || true
    deactivate
    exit \$rc
"

# 5c. FP8 only
run_step "05c" "bench_fp8" bash -c "
    source vllm_venv/bin/activate
    vllm serve '$FP8_MODEL' \
        --port $PORT \
        --dtype auto \
        &
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
    pid=\$(lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null || true)
    [[ -n \"\$pid\" ]] && kill \"\$pid\" 2>/dev/null || true
    deactivate
    exit \$rc
"

# 5d. FP8 + speculative decoding (draft_tokens=1)
run_step "05d" "bench_fp8_spec" bash -c "
    source vllm_venv/bin/activate
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
    pid=\$(lsof -ti :$PORT -sTCP:LISTEN 2>/dev/null || true)
    [[ -n \"\$pid\" ]] && kill \"\$pid\" 2>/dev/null || true
    deactivate
    exit \$rc
"

# ── Step 06: draft token sweep (FP8 + spec) ──────────────────────────────────
run_step "06" "tune_draft_tokens" bash scripts/06_tune_draft_tokens.sh "$FP8_MODEL" "$DRAFT_HEAD"

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
