#!/bin/bash
# run_option3.sh — Proteus & Full FT cross-domain eval matrix
# Trains both conditions sequentially. Evals all 4 domains after every checkpoint.
# This produces the off-diagonal retention data missing from Table 3.

chmod +x "$0"
NTFY_TOPIC="${1:-proteus-aman-2026}"
LOG="results/run_option3.log"
STEPS=2000
N_EVAL=200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"

START_TIME=$(date +%s)
RATE=1.99

mkdir -p results
set -eo pipefail

log()    { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

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

STEP_FILE="results/.current_step_opt3"
STATUS_FILE="results/.live_status_opt3.json"

set_step() { CURRENT_STEP="$1"; echo "$1" > "$STEP_FILE"; }
get_step()  { cat "$STEP_FILE" 2>/dev/null || echo "unknown"; }

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
    [[ ! -f /dev/tty ]] && return
    rm -f "$STATUS_FILE"; : > "$STATUS_FILE"
    ( TTY=/dev/tty
      while true; do
        if [[ -s "$STATUS_FILE" ]]; then
            line=$(format_status_line 2>/dev/null)
            if [[ -n "$line" ]]; then
                rows=$(tput lines 2>/dev/null || echo 24)
                printf "\0337\033[%d;0H\033[2K\033[1;36m%s\033[0m\0338" "$rows" "$line" > "$TTY"
            fi
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
    if [[ -f /dev/tty ]]; then
        rows=$(tput lines 2>/dev/null || echo 24)
        printf "\0337\033[%d;0H\033[2K\0338" "$rows" > /dev/tty
    fi
    rm -f "$STATUS_FILE"
}

on_error() {
    local exit_code=$? line=$1 step
    step=$(get_step)
    stop_status_renderer
    log "CRASH at line $line (exit $exit_code) during: $step"
    notify "Proteus Option3 CRASHED" \
"Step: $step
Exit: $exit_code | Elapsed: $(elapsed_str) | Spent: $(credit_used)" "urgent" "rotating_light"
    kill "${HEARTBEAT_PID:-}" 2>/dev/null || true
    exit $exit_code
}

trap 'on_error $LINENO' ERR

heartbeat() {
    while true; do
        sleep 1800
        notify "Option3 heartbeat" \
"Step: $(get_step)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"
    done
}

heartbeat &
HEARTBEAT_PID=$!
trap 'stop_status_renderer; kill ${HEARTBEAT_PID:-} 2>/dev/null' EXIT

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
run_train() {
    local domain="$1" condition="$2" out_dir="$3"
    shift 3
    set_step "train/$condition/$domain"
    log "START $CURRENT_STEP -> $out_dir $*"
    start_status_renderer
    python train.py --domain "$domain" --condition "$condition" \
        --out_dir "$out_dir" \
        --batch_size 16 --grad_accum 1 \
        --status_file "$STATUS_FILE" "$@" 2>&1 | tee -a "$LOG"
    stop_status_renderer
    log "DONE  $CURRENT_STEP"
}

# Evals all 4 domains against a checkpoint. This is the full cross-domain row.
run_eval_all() {
    local checkpoint="$1" label="$2"
    set_step "eval/$label"
    log "START $CURRENT_STEP (all 4 domains)"
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

# ── Kill stale processes and free VRAM ───────────────────────────────────────
log "Cleaning up stale Python processes..."
for pid in $(pgrep -f "python train.py" 2>/dev/null); do
    log "  Killing stale PID $pid"; kill -9 "$pid" 2>/dev/null || true
done
for pid in $(pgrep -f "python eval.py" 2>/dev/null); do
    log "  Killing stale PID $pid"; kill -9 "$pid" 2>/dev/null || true
done
sleep 2
log "VRAM state after cleanup:"
rocm-smi --showmeminfo vram 2>&1 | tee -a "$LOG" || true
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────
log "=============================="
log "Proteus Option 3 — cross-domain eval matrix"
log "Conditions: proteus + full"
log "Steps: $STEPS | Eval samples: $N_EVAL | ntfy: $NTFY_TOPIC"
log "=============================="

notify "Option3 started" "proteus + full, 4-domain cross-eval | Topic: $NTFY_TOPIC" "default" "rocket"

# ══════════════════════════════════════════════
# PROTEUS chain — 4 domains sequential
# Evals all 4 domains at every checkpoint
# ══════════════════════════════════════════════
log "--- CHAIN: proteus ---"

run_train medical proteus checkpoints/opt3_proteus/medical --max_steps "$STEPS"
run_eval_all checkpoints/opt3_proteus/medical opt3_proteus_after_medical

run_train legal proteus checkpoints/opt3_proteus/legal --max_steps "$STEPS" \
    --start_from checkpoints/opt3_proteus/medical
run_eval_all checkpoints/opt3_proteus/legal opt3_proteus_after_legal

run_train code proteus checkpoints/opt3_proteus/code --max_steps "$STEPS" \
    --start_from checkpoints/opt3_proteus/legal
run_eval_all checkpoints/opt3_proteus/code opt3_proteus_after_code

run_train multilingual proteus checkpoints/opt3_proteus/multilingual --max_steps "$STEPS" \
    --start_from checkpoints/opt3_proteus/code
run_eval_all checkpoints/opt3_proteus/multilingual opt3_proteus_after_multilingual

notify "Proteus chain done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"

# ══════════════════════════════════════════════
# FULL FT chain — 4 domains sequential
# Evals all 4 domains at every checkpoint
# ══════════════════════════════════════════════
log "--- CHAIN: full ---"

run_train medical full checkpoints/opt3_full/medical --max_steps "$STEPS"
run_eval_all checkpoints/opt3_full/medical opt3_full_after_medical

run_train legal full checkpoints/opt3_full/legal --max_steps "$STEPS" \
    --start_from checkpoints/opt3_full/medical
run_eval_all checkpoints/opt3_full/legal opt3_full_after_legal

run_train code full checkpoints/opt3_full/code --max_steps "$STEPS" \
    --start_from checkpoints/opt3_full/legal
run_eval_all checkpoints/opt3_full/code opt3_full_after_code

run_train multilingual full checkpoints/opt3_full/multilingual --max_steps "$STEPS" \
    --start_from checkpoints/opt3_full/code
run_eval_all checkpoints/opt3_full/multilingual opt3_full_after_multilingual

notify "Full FT chain done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "default" "white_check_mark"

# ─────────────────────────────────────────────
log "=============================="
log "Option 3 complete. Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "Option3 COMPLETE" \
"Elapsed: $(elapsed_str) | Total: $(credit_used)

$(eval_summary)" "high" "checkered_flag"

log "Done."