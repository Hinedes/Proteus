#!/bin/bash
# run_missing.sh — Fill missing baselines on the abstract‑to‑title dataset
# Usage: bash run_missing.sh <ntfy_topic>

set -eo pipefail

NTFY_TOPIC="${1:-proteus-aman-2026}"
LOG="results/run_missing.log"
STEPS=2000
N_EVAL=100                                   # Must match your current eval_log.jsonl
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
    printf "%dh %dm" $((secs/3600)) $(( (secs%3600)/60 ))
}

credit_used() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "\$%.2f" "$(awk "BEGIN {printf \"%.2f\", $secs / 3600 * $RATE}")"
}

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

# Clean stale processes and reclaim VRAM
log "Cleaning up any leftover Python processes..."
pkill -f "python train.py" 2>/dev/null || true
pkill -f "python eval.py" 2>/dev/null || true
sleep 2
rocm-smi --showmeminfo vram 2>&1 | tee -a "$LOG" || true

log "=============================="
log "Missing baselines – starting"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="

notify "Missing baselines started" "Topic: $NTFY_TOPIC" "default" "rocket"

# ─────────────────────────────────────────
# 1. LoRA (attention, r=64)
# ─────────────────────────────────────────
log "=== LoRA canonical chain ==="

run_train medical lora checkpoints/lora_canon/medical
run_eval  checkpoints/lora_canon/medical lora_canon_after_medical

run_train legal lora checkpoints/lora_canon/legal \
    --start_from checkpoints/lora_canon/medical
run_eval  checkpoints/lora_canon/legal lora_canon_after_legal

run_train code lora checkpoints/lora_canon/code \
    --start_from checkpoints/lora_canon/legal
run_eval  checkpoints/lora_canon/code lora_canon_after_code

run_train multilingual lora checkpoints/lora_canon/multilingual \
    --start_from checkpoints/lora_canon/code
run_eval  checkpoints/lora_canon/multilingual lora_canon_after_multilingual

# ─────────────────────────────────────────
# 2. LoRA-FFN (FFN, r=64)
# ─────────────────────────────────────────
log "=== LoRA-FFN canonical chain ==="

run_train medical lora_ffn checkpoints/lora_ffn_canon/medical
run_eval  checkpoints/lora_ffn_canon/medical lora_ffn_canon_after_medical

run_train legal lora_ffn checkpoints/lora_ffn_canon/legal \
    --start_from checkpoints/lora_ffn_canon/medical
run_eval  checkpoints/lora_ffn_canon/legal lora_ffn_canon_after_legal

run_train code lora_ffn checkpoints/lora_ffn_canon/code \
    --start_from checkpoints/lora_ffn_canon/legal
run_eval  checkpoints/lora_ffn_canon/code lora_ffn_canon_after_code

run_train multilingual lora_ffn checkpoints/lora_ffn_canon/multilingual \
    --start_from checkpoints/lora_ffn_canon/code
run_eval  checkpoints/lora_ffn_canon/multilingual lora_ffn_canon_after_multilingual

# ─────────────────────────────────────────
# 3. MSF + freeze all attention (5a)
# ─────────────────────────────────────────
log "=== Proteus + freeze all attention (5a) ==="

run_train medical proteus checkpoints/attn_freeze/medical \
    --attention freeze
run_eval  checkpoints/attn_freeze/medical attn_freeze_after_medical

run_train legal proteus checkpoints/attn_freeze/legal \
    --start_from checkpoints/attn_freeze/medical --attention freeze
run_eval  checkpoints/attn_freeze/legal attn_freeze_after_legal

run_train code proteus checkpoints/attn_freeze/code \
    --start_from checkpoints/attn_freeze/legal --attention freeze
run_eval  checkpoints/attn_freeze/code attn_freeze_after_code

run_train multilingual proteus checkpoints/attn_freeze/multilingual \
    --start_from checkpoints/attn_freeze/code --attention freeze
run_eval  checkpoints/attn_freeze/multilingual attn_freeze_after_multilingual

# ─────────────────────────────────────────
# 4. MSF + diagonal attention freeze (5c)
# ─────────────────────────────────────────
log "=== Proteus + diagonal attention freeze (5c) ==="

run_train medical proteus checkpoints/attn_diag/medical \
    --attention diagonal
run_eval  checkpoints/attn_diag/medical attn_diag_after_medical

run_train legal proteus checkpoints/attn_diag/legal \
    --start_from checkpoints/attn_diag/medical --attention diagonal
run_eval  checkpoints/attn_diag/legal attn_diag_after_legal

run_train code proteus checkpoints/attn_diag/code \
    --start_from checkpoints/attn_diag/legal --attention diagonal
run_eval  checkpoints/attn_diag/code attn_diag_after_code

run_train multilingual proteus checkpoints/attn_diag/multilingual \
    --start_from checkpoints/attn_diag/code --attention diagonal
run_eval  checkpoints/attn_diag/multilingual attn_diag_after_multilingual

# ─────────────────────────────────────────
# 5. (Optional) Second seeds for main methods
# ─────────────────────────────────────────
# If credits permit, uncomment the blocks below.
# log "=== Second seed – Full FT ==="
# run_train medical full checkpoints/full_seed2/medical --seed 42
# run_eval  checkpoints/full_seed2/medical full_seed2_after_medical
# ...

log "=============================="
log "All missing baselines completed."
log "Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "Missing baselines COMPLETE" \
"Elapsed: $(elapsed_str) | Total: $(credit_used)

$(python3 -c '
import json
from pathlib import Path
log = Path("results/eval_log.jsonl")
if log.exists():
    for line in log.read_text().splitlines():
        if not line.strip(): continue
        e = json.loads(line)
        d = e.get("domains", {})
        print(f"{e[\"label\"]}: med={d.get(\"medical\",\"?\")} leg={d.get(\"legal\",\"?\")} cod={d.get(\"code\",\"?\")} mul={d.get(\"multilingual\",\"?\")}")
')" "high" "checkered_flag"

log "Done."