#!/bin/bash
# Proteus — EWC + Replay clean rerun (1x MI300X)
# Runs EWC chain then Replay chain sequentially.
# Effective batch size = 16 (matches canonical runs).
#
# Usage: bash run_ewc_replay.sh [ntfy_topic]
chmod +x "$0"

NTFY_TOPIC="${1:-proteus-aman-2026}"
LOG="results/run_ewc_replay.log"
STEPS=500
N_EVAL=200
BS=16

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"

START_TIME=$(date +%s)
RATE=1.99

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
run_train() {
    local domain="$1" condition="$2" out_dir="$3"
    shift 3
    log "START train/$condition/$domain -> $out_dir"
    python train.py \
        --domain "$domain" --condition "$condition" \
        --out_dir "$out_dir" \
        --batch_size $BS --grad_accum 1 \
        --max_steps $STEPS \
        "$@" 2>&1 | tee -a "$LOG"
    log "DONE  train/$condition/$domain"
}

run_eval() {
    local checkpoint="$1" label="$2"
    log "START eval/$label"
    python eval.py --checkpoint "$checkpoint" --label "$label" \
        --n_samples $N_EVAL 2>&1 | tee -a "$LOG"
    log "DONE  eval/$label"
}

build_replay() {
    local domain="$1"
    log "Building replay buffer: $domain"
    python build_replay_buffer.py --domain "$domain" 2>&1 | tee -a "$LOG"
}

# ─────────────────────────────────────────────────────────────────────────────
log "=============================="
log "Proteus — EWC + Replay rerun (1x MI300X)"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="
notify "EWC+Replay rerun started" "1x MI300X | ETA ~2.5h | Topic: $NTFY_TOPIC" "default" "rocket"

# Wipe stale replay buffer
rm -f data/replay_buffer.jsonl
log "Cleared stale replay buffer."

# ══════════════════════════════════════════════
# EWC CHAIN
# Domain 1: no --ewc_state (first domain, penalty off)
# Domain 2+: --ewc_state from previous domain, lambda=1000
# ══════════════════════════════════════════════
log "====== EWC CHAIN ======"

run_train medical ewc checkpoints/ewc_v2/medical
run_eval  checkpoints/ewc_v2/medical ewc_v2_after_medical
notify "EWC medical done" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

run_train legal ewc checkpoints/ewc_v2/legal \
    --start_from checkpoints/ewc_v2/medical \
    --ewc_state  checkpoints/ewc_v2/medical/fisher.pt \
    --ewc_lambda 5000 --ewc_samples 128
run_eval  checkpoints/ewc_v2/legal ewc_v2_after_legal
notify "EWC legal done" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

run_train code ewc checkpoints/ewc_v2/code \
    --start_from checkpoints/ewc_v2/legal \
    --ewc_state  checkpoints/ewc_v2/legal/fisher.pt \
    --ewc_lambda 5000 --ewc_samples 128
run_eval  checkpoints/ewc_v2/code ewc_v2_after_code
notify "EWC code done" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

run_train multilingual ewc checkpoints/ewc_v2/multilingual \
    --start_from checkpoints/ewc_v2/code \
    --ewc_state  checkpoints/ewc_v2/code/fisher.pt \
    --ewc_lambda 5000 --ewc_samples 128
run_eval  checkpoints/ewc_v2/multilingual ewc_v2_after_multilingual
notify "EWC chain COMPLETE" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "high" "white_check_mark"

# ══════════════════════════════════════════════
# REPLAY CHAIN
# Domain 1: no buffer yet
# Domain 2+: growing replay buffer
# ══════════════════════════════════════════════
log "====== REPLAY CHAIN ======"

run_train medical replay checkpoints/replay_v2/medical
run_eval  checkpoints/replay_v2/medical replay_v2_after_medical
build_replay medical
notify "Replay medical done" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

run_train legal replay checkpoints/replay_v2/legal \
    --start_from checkpoints/replay_v2/medical \
    --replay_buffer data/replay_buffer.jsonl
run_eval  checkpoints/replay_v2/legal replay_v2_after_legal
build_replay legal
notify "Replay legal done" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

run_train code replay checkpoints/replay_v2/code \
    --start_from checkpoints/replay_v2/legal \
    --replay_buffer data/replay_buffer.jsonl
run_eval  checkpoints/replay_v2/code replay_v2_after_code
build_replay code
notify "Replay code done" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

run_train multilingual replay checkpoints/replay_v2/multilingual \
    --start_from checkpoints/replay_v2/code \
    --replay_buffer data/replay_buffer.jsonl
run_eval  checkpoints/replay_v2/multilingual replay_v2_after_multilingual
notify "Replay chain COMPLETE" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "high" "white_check_mark"

# ─────────────────────────────────────────────
log "=============================="
log "All done. Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="
notify "EWC+Replay ALL DONE" \
    "Both chains complete.\nElapsed: $(elapsed_str) | Total: $(credit_used)" \
    "high" "checkered_flag"
