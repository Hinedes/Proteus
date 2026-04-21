#!/bin/bash
# Proteus — Full Experiment Runner
# Runs all remaining conditions sequentially, evals after each domain pair,
# logs everything to results/run_all.log, and sends ntfy notification on finish.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh YOUR_NTFY_TOPIC
#
# ntfy: https://ntfy.sh -- free, no account needed.
# Pick any topic name (e.g. "proteus-aman-2026"). Keep it non-obvious.

set -euo pipefail

NTFY_TOPIC="${1:-proteus-notify}"
LOG="results/run_all.log"
STEPS=500
N_EVAL=100

mkdir -p results

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

notify() {
    local title="$1"
    local message="$2"
    local priority="${3:-default}"
    curl -s \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -d "$message" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null || true
}

run_train() {
    local domain="$1"
    local condition="$2"
    shift 2
    log "START train -- condition=$condition domain=$domain $*"
    python train.py --domain "$domain" --condition "$condition" \
        --max_steps "$STEPS" "$@" 2>&1 | tee -a "$LOG"
    log "DONE  train -- condition=$condition domain=$domain"
}

run_eval() {
    local checkpoint="$1"
    local label="$2"
    log "START eval -- $label"
    python eval.py --checkpoint "$checkpoint" --label "$label" \
        --n_samples "$N_EVAL" 2>&1 | tee -a "$LOG"
    log "DONE  eval -- $label"
}

log "=============================="
log "Proteus full experiment run"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="

notify "Proteus started" "All conditions queued. ETA ~2 hours." "default"

# ─────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────
log "--- CONDITION: lora ---"
run_train medical lora
run_train legal   lora
run_eval checkpoints/lora/legal lora_after_legal_v4

notify "Proteus: LoRA done" "$(tail -6 $LOG)" "default"

# ─────────────────────────────────────────────
# EWC
# ─────────────────────────────────────────────
log "--- CONDITION: ewc ---"
run_train medical ewc
run_train legal   ewc --ewc_state checkpoints/ewc/medical/fisher.pt
run_eval checkpoints/ewc/legal ewc_after_legal_v4

notify "Proteus: EWC done" "$(tail -6 $LOG)" "default"

# ─────────────────────────────────────────────
# Replay
# ─────────────────────────────────────────────
log "--- CONDITION: replay ---"
run_train medical replay
python build_replay_buffer.py --domain medical 2>&1 | tee -a "$LOG"
run_train legal replay --replay_buffer data/replay_buffer.jsonl
run_eval checkpoints/replay/legal replay_after_legal_v4

notify "Proteus: Replay done" "$(tail -6 $LOG)" "default"

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
log "=============================="
log "All conditions complete."
log "Results in results/eval_log.jsonl"
log "=============================="

# Pull final summary from eval log
SUMMARY=$(python3 - <<'EOF'
import json
from pathlib import Path

log = Path("results/eval_log.jsonl")
if not log.exists():
    print("No eval log found.")
else:
    entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    for e in entries:
        domains = e.get("domains", {})
        med = domains.get("medical", "?")
        leg = domains.get("legal", "?")
        print(f"{e['label']}: medical={med} legal={leg}")
EOF
)

notify "Proteus DONE" "$SUMMARY" "high"

log "ntfy notification sent."
log "Done."
