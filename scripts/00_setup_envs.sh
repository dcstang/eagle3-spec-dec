#!/usr/bin/env bash
# Task 1: Create the three isolated virtual environments and install dependencies.
# Run once on a fresh H100 node before any other script.
set -uo pipefail   # no -e: run_all.sh owns retry/abort logic

# System deps: python3.12-dev needed for triton/inductor gcc compilation step
sudo apt-get install -y python3.12-dev

PYTHON=python3.12

# ── speculators_venv ─────────────────────────────────────────────────────────
$PYTHON -m venv speculators_venv
source speculators_venv/bin/activate

pip install --upgrade pip
# Safe to re-run: skip clone if already present
[ -d speculators_repo ] || git clone --branch v0.5.0 https://github.com/vllm-project/speculators.git speculators_repo
pip install -e speculators_repo
# speculators does not always pull vllm transitively; install explicitly
pip install vllm==0.20.0 "fastapi<0.137"
deactivate

# ── vllm_venv ────────────────────────────────────────────────────────────────
$PYTHON -m venv vllm_venv
source vllm_venv/bin/activate

pip install --upgrade pip
pip install vllm==0.20.0 "fastapi<0.137" datasets
deactivate

# ── comp_venv ────────────────────────────────────────────────────────────────
$PYTHON -m venv comp_venv
source comp_venv/bin/activate

pip install --upgrade pip
pip install llmcompressor==0.12.0
deactivate

echo "All environments ready: speculators_venv  vllm_venv  comp_venv"
