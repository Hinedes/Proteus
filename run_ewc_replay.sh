#!/bin/bash
# Proteus — EWC + Replay speedrun
# 8x MI300X split 4+4: EWC on GPUs 0-3, Replay on GPUs 4-7
# Both chains run in parallel at each domain step.
# Effective batch size = 4 GPUs x bs4 = 16 (matches canonical runs).
#
# Usage: bash run_ewc_replay.sh [ntfy_topic]
chmod +x "$0"

NTFY_TOPIC="${1:-proteus-aman-2026}"
LOG="results/run_ewc_replay.log"
STEPS=2000
N_EVAL=200
BS=4
N_GPU=4
PORT_EWC=29500
PORT_REPLAY=29501

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"

START_TIME=$(date +%s)
RATE=15.92

mkdir -p results
set -eo pipefail

log()    { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
elapsed_str() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "%dh %02dm" $((secs/3600)) $(( (secs%3600)/60 ))
}
credit_used() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "\$%.2f" "$(awk "BEGIN {printf \"%.2f\", $secs / 3600 * $RATE}")"
}

notify() {
    local title="$1" message="$2" priority="${3:-default}" tags="${4:-}"
    curl -fsS --connect-timeout 10 --max-time 20 \
        -H "Title: $title" -H "Priority: $priority" -H "Markdown: yes" \
        ${tags:+-H "Tags: $tags"} --data-raw "$message" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
}

on_error() {
    local exit_code=$? line=$1
    log "CRASH at line $line (exit $exit_code) | Elapsed: $(elapsed_str) | Spent: $(credit_used)"
    notify "Proteus EWC/Replay CRASHED" \
        "Line: $line | Exit: $exit_code | Elapsed: $(elapsed_str) | Spent: $(credit_used)" \
        "urgent" "rotating_light"
    exit $exit_code
}
trap 'on_error $LINENO' ERR
trap 'log "Exiting. Elapsed: $(elapsed_str) | Spent: $(credit_used)"' EXIT

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

# Run a single training job on a specific GPU set.
# Args: gpus port domain condition out_dir [extra train.py args...]
run_train_on() {
    local gpus="$1" port="$2" domain="$3" condition="$4" out_dir="$5"
    shift 5
    log "START train/$condition/$domain | GPUs=$gpus"
    # Set all common GPU visibility vars for ROCm + cross-vendor portability.
    CUDA_VISIBLE_DEVICES="$gpus" \
    HIP_VISIBLE_DEVICES="$gpus" \
    ROCR_VISIBLE_DEVICES="$gpus" \
    torchrun --nproc_per_node=$N_GPU --master_port=$port \
        train.py \
        --domain "$domain" --condition "$condition" \
        --out_dir "$out_dir" \
        --batch_size $BS --grad_accum 1 \
        "$@" 2>&1 | tee -a "$LOG"
    log "DONE  train/$condition/$domain"
}

# Run two training jobs in parallel. Blocks until BOTH finish.
# Args: gpus1 port1 domain1 condition1 out1 [extra1...] --- gpus2 port2 domain2 condition2 out2 [extra2...]
run_parallel() {
    # Split args on ---
    local -a left=() right=()
    local side="left"
    for arg in "$@"; do
        if [[ "$arg" == "---" ]]; then side="right"; continue; fi
        if [[ "$side" == "left" ]]; then left+=("$arg"); else right+=("$arg"); fi
    done

    local gpus1="${left[0]}" port1="${left[1]}" domain1="${left[2]}" condition1="${left[3]}" out1="${left[4]}"
    local extra1=("${left[@]:5}")
    local gpus2="${right[0]}" port2="${right[1]}" domain2="${right[2]}" condition2="${right[3]}" out2="${right[4]}"
    local extra2=("${right[@]:5}")

    log "PARALLEL: $condition1/$domain1 (GPUs $gpus1) + $condition2/$domain2 (GPUs $gpus2)"

    run_train_on "$gpus1" "$port1" "$domain1" "$condition1" "$out1" "${extra1[@]}" &
    local pid1=$!
    run_train_on "$gpus2" "$port2" "$domain2" "$condition2" "$out2" "${extra2[@]}" &
    local pid2=$!

    wait $pid1 || { log "FAILED: $condition1/$domain1"; exit 1; }
    wait $pid2 || { log "FAILED: $condition2/$domain2"; exit 1; }
}

run_eval() {
    local checkpoint="$1" label="$2"
    log "START eval/$label"
    python eval.py --checkpoint "$checkpoint" --label "$label" --n_samples $N_EVAL 2>&1 | tee -a "$LOG"
    log "DONE  eval/$label"
}

build_replay() {
    local domain="$1"
    log "Building replay buffer for $domain..."
    python build_replay_buffer.py --domain "$domain" 2>&1 | tee -a "$LOG"
}

# ─────────────────────────────────────────────────────────────────────────────
log "=============================="
log "Proteus — EWC + Replay speedrun (8x MI300X, 4+4 split)"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="

notify "EWC+Replay speedrun started" \
    "8x MI300X | 4+4 GPU split | ETA ~30min\nTopic: $NTFY_TOPIC" "default" "rocket"

# Wipe any stale replay buffer from previous broken runs
rm -f data/replay_buffer.jsonl
log "Cleared stale replay buffer."

# ══════════════════════════════════════════════
# DOMAIN 1: medical
# EWC: no prior state (first domain, penalty disabled)
# Replay: no buffer yet (first domain)
# ══════════════════════════════════════════════
log "--- DOMAIN 1: medical ---"
run_parallel \
    "0,1,2,3" $PORT_EWC    medical ewc    checkpoints/ewc_v2/medical    --max_steps $STEPS \
    --- \
    "4,5,6,7" $PORT_REPLAY medical replay checkpoints/replay_v2/medical --max_steps $STEPS

# Eval both
run_eval checkpoints/ewc_v2/medical    ewc_v2_after_medical
run_eval checkpoints/replay_v2/medical replay_v2_after_medical

# Build replay buffer from medical
build_replay medical

notify "Domain 1 (medical) done" \
    "EWC + Replay | Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

# ══════════════════════════════════════════════
# DOMAIN 2: legal
# EWC: load fisher from medical
# Replay: use buffer (medical samples)
# ══════════════════════════════════════════════
log "--- DOMAIN 2: legal ---"
run_parallel \
    "0,1,2,3" $PORT_EWC    legal ewc    checkpoints/ewc_v2/legal    --max_steps $STEPS \
        --start_from checkpoints/ewc_v2/medical \
        --ewc_state  checkpoints/ewc_v2/medical/fisher.pt \
        --ewc_lambda 1000 --ewc_samples 128 \
    --- \
    "4,5,6,7" $PORT_REPLAY legal replay checkpoints/replay_v2/legal --max_steps $STEPS \
        --start_from checkpoints/replay_v2/medical \
        --replay_buffer data/replay_buffer.jsonl

run_eval checkpoints/ewc_v2/legal    ewc_v2_after_legal
run_eval checkpoints/replay_v2/legal replay_v2_after_legal

build_replay legal

notify "Domain 2 (legal) done" \
    "EWC + Replay | Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

# ══════════════════════════════════════════════
# DOMAIN 3: code
# ══════════════════════════════════════════════
log "--- DOMAIN 3: code ---"
run_parallel \
    "0,1,2,3" $PORT_EWC    code ewc    checkpoints/ewc_v2/code    --max_steps $STEPS \
        --start_from checkpoints/ewc_v2/legal \
        --ewc_state  checkpoints/ewc_v2/legal/fisher.pt \
        --ewc_lambda 1000 --ewc_samples 128 \
    --- \
    "4,5,6,7" $PORT_REPLAY code replay checkpoints/replay_v2/code --max_steps $STEPS \
        --start_from checkpoints/replay_v2/legal \
        --replay_buffer data/replay_buffer.jsonl

run_eval checkpoints/ewc_v2/code    ewc_v2_after_code
run_eval checkpoints/replay_v2/code replay_v2_after_code

build_replay code

notify "Domain 3 (code) done" \
    "EWC + Replay | Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

# ══════════════════════════════════════════════
# DOMAIN 4: multilingual
# ══════════════════════════════════════════════
log "--- DOMAIN 4: multilingual ---"
run_parallel \
    "0,1,2,3" $PORT_EWC    multilingual ewc    checkpoints/ewc_v2/multilingual    --max_steps $STEPS \
        --start_from checkpoints/ewc_v2/code \
        --ewc_state  checkpoints/ewc_v2/code/fisher.pt \
        --ewc_lambda 1000 --ewc_samples 128 \
    --- \
    "4,5,6,7" $PORT_REPLAY multilingual replay checkpoints/replay_v2/multilingual --max_steps $STEPS \
        --start_from checkpoints/replay_v2/code \
        --replay_buffer data/replay_buffer.jsonl

run_eval checkpoints/ewc_v2/multilingual    ewc_v2_after_multilingual
run_eval checkpoints/replay_v2/multilingual replay_v2_after_multilingual

# ─────────────────────────────────────────────
log "=============================="
log "All done. Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "EWC+Replay COMPLETE" \
    "All 4 domains done.\nElapsed: $(elapsed_str) | Total: $(credit_used)" \
    "high" "checkered_flag"
