#!/bin/bash
chmod +x "$0"
# Proteus — Full Experimental Runner
#
# Section 1: Extended Medical→Legal (4000 steps, 2 conditions)
#   Stress-tests forgetting harder. Widens Table 2 gap.
#   Saves to checkpoints/{condition}_4k/ — does NOT touch canonical checkpoints.
#
# Section 2: Canonical 4-domain sequential chains (2000 steps, 2 conditions)
#   Single clean run for both Proteus and Full. Fixes Table 3 chimera.
#   Saves to checkpoints/{condition}_canon/ — isolated from prior runs.
#
# Section 3: Attention sweeps (2000 steps, 2 variants, 4 domains each)
#   5a: --attention freeze   → proteus_attn_freeze
#   5c: --attention diagonal → proteus_attn_diagonal
#   Sequential chains, start from base model.
#
# Usage:
#   ./run_all.sh YOUR_NTFY_TOPIC

NTFY_TOPIC="${1:-proteus-aman-2026}"
LOG="results/run_all.log"
STEPS=2000
LONG_STEPS=4000
N_EVAL=100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"

# ── TunableOp: GEMM kernel tuning for MI300X ──────────────────────────────────
# One-time tuning pass selects the fastest hipBLASLt/rocBLAS kernel per GEMM
# shape. CSV is reused by all subsequent runs at zero overhead.
# Expected gain: 5-22% throughput improvement on LLM workloads.
# TUNABLEOP_CSV="results/tunableop_results.csv"
# export PYTORCH_TUNABLEOP_ENABLED=1
# export PYTORCH_TUNABLEOP_FILENAME="$TUNABLEOP_CSV"
START_TIME=$(date +%s)
RATE=1.99   # USD/hr for MI300X

mkdir -p results

set -eo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

notify() {
    local title="$1"
    local message="$2"
    local priority="${3:-default}"
    local tags="${4:-}"
    if ! curl -fsS --connect-timeout 10 --max-time 20 \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -H "Markdown: yes" \
        ${tags:+-H "Tags: $tags"} \
        --data-raw "$message" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null; then
        log "[notify] WARNING: failed to send to topic '$NTFY_TOPIC' (title: $title)"
    fi
}

elapsed_str() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "%dh %02dm" $((secs/3600)) $(( (secs%3600)/60 ))
}

credit_used() {
    local secs=$(( $(date +%s) - START_TIME ))
    printf "\$%.2f" "$(awk "BEGIN {printf \"%.2f\", $secs / 3600 * $RATE}")"
}

STEP_FILE="results/.current_step"
STATUS_FILE="results/.live_status.json"
CURRENT_STEP="unknown"
STATUS_RENDER_PID=""

set_step() {
    CURRENT_STEP="$1"
    echo "$1" > "$STEP_FILE"
}

get_step() {
    cat "$STEP_FILE" 2>/dev/null || echo "unknown"
}

last_clean_log() {
    grep -E "^\[20|loss|perplexity|DONE|START|CRASH" "$LOG" 2>/dev/null \
        | tail -1 \
        || echo "no log yet"
}

format_status_line() {
    python3 - "$STATUS_FILE" <<'PY'
import json, math, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("", end=""); raise SystemExit(0)
def fmt_eta(s):
    if s is None: return "--:--"
    try: secs = int(max(0, float(s)))
    except: return "--:--"
    m, s = divmod(secs, 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
phase = data.get("phase"); state = str(data.get("state",""))
if phase == "train":
    step = int(data.get("step",0) or 0); total = int(data.get("total_steps",0) or 0)
    pct = (100.0*step/total) if total>0 else 0.0
    it_s = float(data.get("it_s",0.0) or 0.0); eta = fmt_eta(data.get("eta_s"))
    loss = data.get("loss")
    loss_txt = f"{float(loss):.4f}" if isinstance(loss,(int,float)) and math.isfinite(float(loss)) else "--"
    print(f"[train {data.get('condition','?')}/{data.get('domain','?')}] {step}/{total} ({pct:5.1f}%) | {it_s:6.2f} it/s | eta {eta} | loss {loss_txt} | {state}", end="")
elif phase == "eval":
    os_= int(data.get("overall_step",0) or 0); ot = int(data.get("overall_total",0) or 0)
    pct = (100.0*os_/ot) if ot>0 else 0.0; it_s = float(data.get("it_s",0.0) or 0.0)
    ppl = data.get("ppl_so_far")
    ppl_txt = f"{float(ppl):.3f}" if isinstance(ppl,(int,float)) and math.isfinite(float(ppl)) else "--"
    print(f"[eval {data.get('domain_index','?')}/{data.get('domains_total','?')} {data.get('domain','?')}] {os_}/{ot} ({pct:5.1f}%) | {it_s:6.2f} it/s | ppl~ {ppl_txt} | {state}", end="")
else:
    print("", end="")
PY
}

start_status_renderer() {
    [[ ! -t 2 ]] && return
    rm -f "$STATUS_FILE"; : > "$STATUS_FILE"
    ( while true; do
        if [[ -s "$STATUS_FILE" ]]; then
            line=$(format_status_line)
            [[ -n "$line" ]] && printf "\r\033[2K%s" "$line" >&2
        fi
        sleep 1
    done ) &
    STATUS_RENDER_PID=$!
}

stop_status_renderer() {
    if [[ -n "${STATUS_RENDER_PID:-}" ]]; then
        kill "$STATUS_RENDER_PID" 2>/dev/null || true
        wait "$STATUS_RENDER_PID" 2>/dev/null || true
        STATUS_RENDER_PID=""
    fi
    [[ -t 2 ]] && printf "\r\033[2K" >&2
    rm -f "$STATUS_FILE"
}

on_error() {
    local exit_code=$? line=$1 step
    step=$(get_step)
    stop_status_renderer
    log "CRASH at line $line (exit $exit_code) during: $step"
    notify "Proteus CRASHED" \
"Step: $step
Exit: $exit_code | Elapsed: $(elapsed_str) | Spent: $(credit_used)
Last: $(last_clean_log)
Resume manually from failed step." "urgent" "rotating_light"
    kill "${HEARTBEAT_PID:-}" 2>/dev/null || true
    exit $exit_code
}

trap 'on_error $LINENO' ERR

heartbeat() {
    while true; do
        sleep 1800
        notify "Proteus heartbeat" \
"Step: $(get_step)
Elapsed: $(elapsed_str) | Spent: $(credit_used)
Last: $(last_clean_log)" "low"
    done
}

heartbeat &
HEARTBEAT_PID=$!
trap 'on_error $LINENO' ERR
trap 'stop_status_renderer; kill ${HEARTBEAT_PID:-} 2>/dev/null' EXIT

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
run_train() {
    # run_train <domain> <condition> <out_dir> [extra args...]
    local domain="$1" condition="$2" out_dir="$3"
    shift 3
    set_step "train/$condition/$domain"
    log "START $CURRENT_STEP -> $out_dir $*"
    start_status_renderer
    python train.py --domain "$domain" --condition "$condition" \
        --out_dir "$out_dir" \
        --batch_size 16 --grad_accum 1 --status_file "$STATUS_FILE" "$@" 2>&1 | tee -a "$LOG"
    stop_status_renderer
    log "DONE  $CURRENT_STEP"
}

run_eval() {
    local checkpoint="$1" label="$2"
    set_step "eval/$label"
    log "START $CURRENT_STEP"
    start_status_renderer
    python eval.py --checkpoint "$checkpoint" --label "$label" \
        --n_samples "$N_EVAL" --status_file "$STATUS_FILE" 2>&1 | tee -a "$LOG"
    stop_status_renderer
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
        print(f"{e['label']}: med={d.get('medical','?')} leg={d.get('legal','?')} cod={d.get('code','?')} mul={d.get('multilingual','?')}")
EOF
}

# ─────────────────────────────────────────────
log "=============================="
log "Proteus full experimental run"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="

# ── TunableOp warmup disabled ────────────────────────────────────────────────
# TunableOp variables and warmup pass are intentionally disabled.

notify "Proteus full run started" \
"Sections: Extended 4k | Canonical chains | Attention sweeps
Topic: $NTFY_TOPIC" "default" "rocket"


# ══════════════════════════════════════════════
# SECTION 1: Extended Medical→Legal (4000 steps)
# ══════════════════════════════════════════════
log "=============================="
log "SECTION 1: Extended 4k Medical→Legal"
log "=============================="

# run_train medical full     checkpoints/full_4k/medical    --max_steps "$LONG_STEPS"
# run_eval  checkpoints/full_4k/medical    full_4k_after_medical
#
# run_train legal   full     checkpoints/full_4k/legal      --max_steps "$LONG_STEPS" \
#     --start_from checkpoints/full_4k/medical
# run_eval  checkpoints/full_4k/legal      full_4k_after_legal
#
# run_train medical proteus  checkpoints/proteus_4k/medical --max_steps "$LONG_STEPS"
# run_eval  checkpoints/proteus_4k/medical proteus_4k_after_medical
#
# run_train legal   proteus  checkpoints/proteus_4k/legal   --max_steps "$LONG_STEPS" \
#     --start_from checkpoints/proteus_4k/medical
# run_eval  checkpoints/proteus_4k/legal   proteus_4k_after_legal
#
# notify "Section 1 done: Extended 4k" \
# "$(eval_summary)
# Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"


# ══════════════════════════════════════════════
# SECTION 2: Canonical 4-domain sequential chains
# ══════════════════════════════════════════════
log "=============================="
log "SECTION 2: Canonical 4-domain chains (2000 steps)"
log "=============================="

# log "--- CHAIN: proteus_canon ---"
# run_train medical    proteus checkpoints/proteus_canon/medical       --max_steps "$STEPS"
# run_eval  checkpoints/proteus_canon/medical       proteus_canon_after_medical
#
# run_train legal      proteus checkpoints/proteus_canon/legal         --max_steps "$STEPS" \
#     --start_from checkpoints/proteus_canon/medical
# run_eval  checkpoints/proteus_canon/legal         proteus_canon_after_legal
#
# run_train code       proteus checkpoints/proteus_canon/code          --max_steps "$STEPS" \
#     --start_from checkpoints/proteus_canon/legal
# run_eval  checkpoints/proteus_canon/code          proteus_canon_after_code
#
# run_train multilingual proteus checkpoints/proteus_canon/multilingual --max_steps "$STEPS" \
#     --start_from checkpoints/proteus_canon/code
# run_eval  checkpoints/proteus_canon/multilingual  proteus_canon_after_multilingual
#
# notify "Proteus canonical chain done" \
# "$(eval_summary)
# Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"
#
# log "--- CHAIN: full_canon ---"
# run_train medical    full checkpoints/full_canon/medical       --max_steps "$STEPS"
# run_eval  checkpoints/full_canon/medical       full_canon_after_medical
#
# run_train legal      full checkpoints/full_canon/legal         --max_steps "$STEPS" \
#     --start_from checkpoints/full_canon/medical
# run_eval  checkpoints/full_canon/legal         full_canon_after_legal
#
# run_train code       full checkpoints/full_canon/code          --max_steps "$STEPS" \
#     --start_from checkpoints/full_canon/legal
# run_eval  checkpoints/full_canon/code          full_canon_after_code
#
# run_train multilingual full checkpoints/full_canon/multilingual --max_steps "$STEPS" \
#     --start_from checkpoints/full_canon/code
# run_eval  checkpoints/full_canon/multilingual  full_canon_after_multilingual
#
# log "--- CHAIN: lora_canon ---"
# run_train medical    lora checkpoints/lora_canon/medical       --max_steps "$STEPS"
# run_eval  checkpoints/lora_canon/medical       lora_canon_after_medical
#
# run_train legal      lora checkpoints/lora_canon/legal         --max_steps "$STEPS" \
#     --start_from checkpoints/lora_canon/medical
# run_eval  checkpoints/lora_canon/legal         lora_canon_after_legal
#
# run_train code       lora checkpoints/lora_canon/code          --max_steps "$STEPS" \
#     --start_from checkpoints/lora_canon/legal
# run_eval  checkpoints/lora_canon/code          lora_canon_after_code
#
# run_train multilingual lora checkpoints/lora_canon/multilingual --max_steps "$STEPS" \
#     --start_from checkpoints/lora_canon/code
# run_eval  checkpoints/lora_canon/multilingual  lora_canon_after_multilingual
#
# notify "LoRA canonical chain done" \
# "$(eval_summary)
# Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"

#log "--- CHAIN: ewc_canon ---"
# run_train medical    ewc checkpoints/ewc_canon/medical       --max_steps "$STEPS"
#run_eval  checkpoints/ewc_canon/medical       ewc_canon_after_medical

run_train legal      ewc checkpoints/ewc_canon/legal         --max_steps "$STEPS" \
    --start_from checkpoints/ewc_canon/medical \
    --ewc_state checkpoints/ewc_canon/medical/fisher.pt
run_eval  checkpoints/ewc_canon/legal         ewc_canon_after_legal

run_train code       ewc checkpoints/ewc_canon/code          --max_steps "$STEPS" \
    --start_from checkpoints/ewc_canon/legal \
    --ewc_state checkpoints/ewc_canon/legal/fisher.pt
run_eval  checkpoints/ewc_canon/code          ewc_canon_after_code

run_train multilingual ewc checkpoints/ewc_canon/multilingual --max_steps "$STEPS" \
    --start_from checkpoints/ewc_canon/code \
    --ewc_state checkpoints/ewc_canon/code/fisher.pt
run_eval  checkpoints/ewc_canon/multilingual  ewc_canon_after_multilingual

notify "EWC canonical chain done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"

log "--- CHAIN: replay_canon ---"
run_train medical    replay checkpoints/replay_canon/medical       --max_steps "$STEPS"
run_eval  checkpoints/replay_canon/medical       replay_canon_after_medical
log "Building replay buffer from medical..."
python build_replay_buffer.py --domain medical 2>&1 | tee -a "$LOG"

run_train legal      replay checkpoints/replay_canon/legal         --max_steps "$STEPS" \
    --start_from checkpoints/replay_canon/medical \
    --replay_buffer data/replay_buffer.jsonl
run_eval  checkpoints/replay_canon/legal         replay_canon_after_legal
log "Building replay buffer: appending legal..."
python build_replay_buffer.py --domain legal 2>&1 | tee -a "$LOG"

run_train code       replay checkpoints/replay_canon/code          --max_steps "$STEPS" \
    --start_from checkpoints/replay_canon/legal \
    --replay_buffer data/replay_buffer.jsonl
run_eval  checkpoints/replay_canon/code          replay_canon_after_code
log "Building replay buffer: appending code..."
python build_replay_buffer.py --domain code 2>&1 | tee -a "$LOG"

run_train multilingual replay checkpoints/replay_canon/multilingual --max_steps "$STEPS" \
    --start_from checkpoints/replay_canon/code \
    --replay_buffer data/replay_buffer.jsonl
run_eval  checkpoints/replay_canon/multilingual  replay_canon_after_multilingual

notify "Section 2 done: Canonical chains" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"


# ══════════════════════════════════════════════
# SECTION 3: Attention sweeps (5a and 5c)
# ══════════════════════════════════════════════
log "=============================="
log "SECTION 3: Attention sweeps"
log "=============================="

log "--- CHAIN: proteus_attn_freeze (5a) ---"
run_train medical    proteus checkpoints/proteus_attn_freeze/medical       --max_steps "$STEPS" --attention freeze
run_eval  checkpoints/proteus_attn_freeze/medical       proteus_freeze_after_medical

run_train legal      proteus checkpoints/proteus_attn_freeze/legal         --max_steps "$STEPS" --attention freeze \
    --start_from checkpoints/proteus_attn_freeze/medical
run_eval  checkpoints/proteus_attn_freeze/legal         proteus_freeze_after_legal

run_train code       proteus checkpoints/proteus_attn_freeze/code          --max_steps "$STEPS" --attention freeze \
    --start_from checkpoints/proteus_attn_freeze/legal
run_eval  checkpoints/proteus_attn_freeze/code          proteus_freeze_after_code

run_train multilingual proteus checkpoints/proteus_attn_freeze/multilingual --max_steps "$STEPS" --attention freeze \
    --start_from checkpoints/proteus_attn_freeze/code
run_eval  checkpoints/proteus_attn_freeze/multilingual  proteus_freeze_after_multilingual

notify "5a done: attn_freeze" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"

log "--- CHAIN: proteus_attn_diagonal (5c) ---"
run_train medical    proteus checkpoints/proteus_attn_diagonal/medical       --max_steps "$STEPS" --attention diagonal
run_eval  checkpoints/proteus_attn_diagonal/medical       proteus_diagonal_after_medical

run_train legal      proteus checkpoints/proteus_attn_diagonal/legal         --max_steps "$STEPS" --attention diagonal \
    --start_from checkpoints/proteus_attn_diagonal/medical
run_eval  checkpoints/proteus_attn_diagonal/legal         proteus_diagonal_after_legal

run_train code       proteus checkpoints/proteus_attn_diagonal/code          --max_steps "$STEPS" --attention diagonal \
    --start_from checkpoints/proteus_attn_diagonal/legal
run_eval  checkpoints/proteus_attn_diagonal/code          proteus_diagonal_after_code

run_train multilingual proteus checkpoints/proteus_attn_diagonal/multilingual --max_steps "$STEPS" --attention diagonal \
    --start_from checkpoints/proteus_attn_diagonal/code
run_eval  checkpoints/proteus_attn_diagonal/multilingual  proteus_diagonal_after_multilingual

notify "Section 3 done: Attention sweeps" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"


# ─────────────────────────────────────────────
log "=============================="
log "All sections complete."
log "Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "Proteus COMPLETE" \
"All 3 sections done.
Elapsed: $(elapsed_str) | Total: $(credit_used)

$(eval_summary)" "high" "checkered_flag"

log "Done."