#!/bin/bash
# run_minimal.sh — Minimal 3‑domain chain for paper salvage (fits ≤ 8 h)
# Usage: bash run_minimal.sh <ntfy_topic>

set -eo pipefail

NTFY_TOPIC="${1:-proteus-aman-2026}"
LOG="results/run_minimal.log"
STEPS=2000
N_EVAL=100   # must match your eval setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"

START_TIME=$(date +%s)
RATE=1.99

mkdir -p results

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

notify() {
    local title="$1" message="$2" priority="${3:-default}" tags="${4:-}"
    curl -fsS --connect-timeout 10 --max-time 20 \
        -H "Title: $title" -H "Priority: $priority" -H "Markdown: yes" \
        ${tags:+-H "Tags: $tags"} --data-raw "$message" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
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
# Helpers (using your existing train/eval scripts)
# ─────────────────────────────────────────────
run_train() {
    local domain="$1" condition="$2" out_dir="$3"; shift 3
    log "START train $condition/$domain → $out_dir"
    python train.py --domain "$domain" --condition "$condition" \
        --out_dir "$out_dir" \
        --batch_size 16 --grad_accum 1 --max_steps "$STEPS" "$@" 2>&1 | tee -a "$LOG"
    log "DONE  train $condition/$domain"
}

run_eval() {
    local checkpoint="$1" label="$2"
    log "START eval $label"
    python eval.py --checkpoint "$checkpoint" --label "$label" \
        --n_samples "$N_EVAL" 2>&1 | tee -a "$LOG"
    log "DONE  eval $label"
}

# ─────────────────────────────────────────────
# Cleanup & initial VRAM audit
# ─────────────────────────────────────────────
log "Cleaning up any leftover Python processes..."
pkill -f "python train.py" 2>/dev/null || true
pkill -f "python eval.py" 2>/dev/null || true
sleep 2
rocm-smi --showmeminfo vram 2>&1 | tee -a "$LOG" || true

# ─────────────────────────────────────────────
# 1. Zero‑shot baseline (fresh base model)
# ─────────────────────────────────────────────
log "=== 1. Zero‑shot baseline ==="
run_eval "google/gemma-4-E4B-it" "baseline_zero_shot"

# ─────────────────────────────────────────────
# 2. Full Fine‑Tune chain (Medical → Legal → Code)
# ─────────────────────────────────────────────
log "=== 2. Full Fine‑Tune chain ==="

run_train medical full checkpoints/full_chain/medical
run_eval  checkpoints/full_chain/medical full_chain_after_medical

run_train legal full checkpoints/full_chain/legal \
    --start_from checkpoints/full_chain/medical
run_eval  checkpoints/full_chain/legal full_chain_after_legal

run_train code full checkpoints/full_chain/code \
    --start_from checkpoints/full_chain/legal
run_eval  checkpoints/full_chain/code full_chain_after_code

# ─────────────────────────────────────────────
# 3. MSF chain (Medical → Legal → Code)
# ─────────────────────────────────────────────
log "=== 3. MSF (proteus) chain ==="

run_train medical proteus checkpoints/msf_chain/medical
run_eval  checkpoints/msf_chain/medical msf_chain_after_medical

run_train legal proteus checkpoints/msf_chain/legal \
    --start_from checkpoints/msf_chain/medical
run_eval  checkpoints/msf_chain/legal msf_chain_after_legal

run_train code proteus checkpoints/msf_chain/code \
    --start_from checkpoints/msf_chain/legal
run_eval  checkpoints/msf_chain/code msf_chain_after_code

# ─────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────
log "=============================="
log "All experiments done."
log "Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "Minimal run COMPLETE" \
"Elapsed: $(elapsed_str) | Spent: $(credit_used)

Results:
$(python3 -c '
import json
from pathlib import Path
log = Path("results/eval_log.jsonl")
if log.exists():
    for line in log.read_text().splitlines():
        if not line.strip(): continue
        e = json.loads(line)
        d = e.get("domains", {})
        print(f"{e[\"label\"]}: med={d.get(\"medical\",\"?\")} leg={d.get(\"legal\",\"?\")} cod={d.get(\"code\",\"?\")}")
')" "high" "checkered_flag"

log "Done."