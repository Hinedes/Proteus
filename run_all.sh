#!/bin/bash
# Proteus — Full Experiment Runner v2
# - Failure detection: notifies immediately on any crash with the error
# - Heartbeat: pings every 30 minutes with current progress
# - Credit estimate: tracks elapsed time and estimated spend
# - Recovery: on crash, logs exactly which command failed and how to resume
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh YOUR_NTFY_TOPIC

NTFY_TOPIC="${1:-proteus-notify}"
LOG="results/run_all.log"
STEPS=500
N_EVAL=100
START_TIME=$(date +%s)
RATE=1.99   # USD/hr for MI300X

mkdir -p results

# Fail on error, and fail if any command in a pipeline fails
set -eo pipefail

# ─────────────────────────────────────────────
# Logging + notifications
# ─────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

notify() {
    local title="$1"
    local message="$2"
    local priority="${3:-default}"
    local tags="${4:-}"
    curl -s \
        -H "Title: $title" \
        -H "Priority: $priority" \
        ${tags:+-H "Tags: $tags"} \
        -d "$message" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null || true
}

elapsed_str() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "%dh %02dm" $((secs/3600)) $(( (secs%3600)/60 ))
}

credit_used() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "\$%.2f" "$(awk "BEGIN {printf \"%.2f\", $secs / 3600 * $RATE}")"
}

# ─────────────────────────────────────────────
# Failure handler
# ─────────────────────────────────────────────
STEP_FILE="results/.current_step"
CURRENT_STEP="unknown"

set_step() {
    CURRENT_STEP="$1"
    echo "$1" > "$STEP_FILE"
}

get_step() {
    cat "$STEP_FILE" 2>/dev/null || echo "unknown"
}

# Clean last meaningful log line (skip tqdm bars and empty lines)
last_clean_log() {
    grep -E "^\[20|loss|perplexity|DONE|START|CRASH" "$LOG" 2>/dev/null \
        | tail -1 \
        || echo "no log yet"
}

on_error() {
    local exit_code=$?
    local line=$1
    local step
    step=$(get_step)
    log "CRASH at line $line (exit $exit_code) during: $step"
    local tail_log
    tail_log=$(last_clean_log)
    notify "Proteus CRASHED" \
"Step: $step
Exit: $exit_code
Elapsed: $(elapsed_str) | Spent: $(credit_used)

Last: $tail_log

Resume: ./run_all.sh $NTFY_TOPIC" \
        "urgent" "rotating_light"
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    exit $exit_code
}

trap 'on_error $LINENO' ERR

# ─────────────────────────────────────────────
# Heartbeat (background process, every 30 min)
# ─────────────────────────────────────────────
heartbeat() {
    while true; do
        sleep 1800
        notify "Proteus heartbeat" \
"Step: $(get_step)
Elapsed: $(elapsed_str) | Spent: $(credit_used)
Last: $(last_clean_log)" \
            "low"
    done
}

heartbeat &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null; on_error $LINENO' ERR
trap 'kill $HEARTBEAT_PID 2>/dev/null' EXIT

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
run_train() {
    local domain="$1"
    local condition="$2"
    shift 2
    set_step "train/$condition/$domain"
    log "START $CURRENT_STEP $*"
    python train.py --domain "$domain" --condition "$condition" \
        --max_steps "$STEPS" --compile "$@" 2>&1 | tee -a "$LOG"
    log "DONE  $CURRENT_STEP"
}

run_eval() {
    local checkpoint="$1"
    local label="$2"
    set_step "eval/$label"
    log "START $CURRENT_STEP"
    python eval.py --checkpoint "$checkpoint" --label "$label" \
        --n_samples "$N_EVAL" 2>&1 | tee -a "$LOG"
    log "DONE  $CURRENT_STEP"
}

eval_summary() {
    python3 - <<'EOF'
import json
from pathlib import Path
log = Path("results/eval_log.jsonl")
if not log.exists():
    print("No results yet.")
else:
    for line in log.read_text().splitlines():
        if not line.strip(): continue
        e = json.loads(line)
        d = e.get("domains", {})
        med = d.get("medical", "?")
        leg = d.get("legal", "?")
        print(f"{e['label']}: med={med} leg={leg}")
EOF
}

# ─────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────
log "=============================="
log "Proteus full experiment run v2"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="

notify "Proteus started" \
"Conditions queued: LoRA, EWC, Replay
ETA ~2 hours | Budget ~\$4
Topic: $NTFY_TOPIC" \
    "default" "rocket"

# ─────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────
log "--- CONDITION: lora ---"
run_train medical lora
run_train legal   lora
run_eval checkpoints/lora/legal lora_after_legal_v4

notify "LoRA done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" \
    "default" "white_check_mark"

# ─────────────────────────────────────────────
# EWC
# ─────────────────────────────────────────────
log "--- CONDITION: ewc ---"
run_train medical ewc
run_train legal   ewc --ewc_state checkpoints/ewc/medical/fisher.pt
run_eval checkpoints/ewc/legal ewc_after_legal_v4

notify "EWC done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" \
    "default" "white_check_mark"

# ─────────────────────────────────────────────
# Replay
# ─────────────────────────────────────────────
log "--- CONDITION: replay ---"
run_train medical replay

CURRENT_STEP="build_replay_buffer/medical"
set_step "build_replay_buffer/medical"
log "START $CURRENT_STEP"
python build_replay_buffer.py --domain medical 2>&1 | tee -a "$LOG"
log "DONE  $CURRENT_STEP"

run_train legal replay --replay_buffer data/replay_buffer.jsonl
run_eval checkpoints/replay/legal replay_after_legal_v4

notify "Replay done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" \
    "default" "white_check_mark"

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
log "=============================="
log "All conditions complete."
log "Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "Proteus COMPLETE" \
"All conditions done!
Elapsed: $(elapsed_str) | Total: $(credit_used)

$(eval_summary)" \
    "high" "checkered_flag"

log "Done."