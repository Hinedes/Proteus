#!/bin/bash
chmod +x "$0"
NTFY_TOPIC="${1:-proteus-aman-2026}"
LOG="results/run_all.log"
STEPS=2000
N_EVAL=100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"

START_TIME=$(date +%s)
RATE=1.99

mkdir -p results
set -eo pipefail

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

STEP_FILE="results/.current_step"
STATUS_FILE="results/.live_status.json"
CURRENT_STEP="unknown"
STATUS_RENDER_PID=""

set_step() { CURRENT_STEP="$1"; echo "$1" > "$STEP_FILE"; }
get_step()  { cat "$STEP_FILE" 2>/dev/null || echo "unknown"; }

last_clean_log() {
    grep -E "^\[20|loss|perplexity|DONE|START|CRASH" "$LOG" 2>/dev/null \
        | tail -1 || echo "no log yet"
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
    notify "Proteus CRASHED" \
"Step: $step
Exit: $exit_code | Elapsed: $(elapsed_str) | Spent: $(credit_used)
Last: $(last_clean_log)" "urgent" "rotating_light"
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

log "=============================="
log "Proteus — LoRA-FFN ablation"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="

notify "LoRA-FFN ablation started" "Topic: $NTFY_TOPIC" "default" "rocket"

# ══════════════════════════════════════════════
# LoRA-FFN: 4-domain canonical chain
# Same location as Proteus (gate/up/down proj), low-rank mechanism
# ══════════════════════════════════════════════
log "--- CHAIN: lora_ffn_canon ---"

run_train medical lora_ffn checkpoints/lora_ffn_canon/medical --max_steps "$STEPS"
run_eval  checkpoints/lora_ffn_canon/medical lora_ffn_canon_after_medical

run_train legal lora_ffn checkpoints/lora_ffn_canon/legal --max_steps "$STEPS" \
    --start_from checkpoints/lora_ffn_canon/medical
run_eval  checkpoints/lora_ffn_canon/legal lora_ffn_canon_after_legal

run_train code lora_ffn checkpoints/lora_ffn_canon/code --max_steps "$STEPS" \
    --start_from checkpoints/lora_ffn_canon/legal
run_eval  checkpoints/lora_ffn_canon/code lora_ffn_canon_after_code

run_train multilingual lora_ffn checkpoints/lora_ffn_canon/multilingual --max_steps "$STEPS" \
    --start_from checkpoints/lora_ffn_canon/code
run_eval  checkpoints/lora_ffn_canon/multilingual lora_ffn_canon_after_multilingual

notify "LoRA-FFN chain done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "high" "white_check_mark"

# ─────────────────────────────────────────────
log "=============================="
log "All done. Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "LoRA-FFN COMPLETE" \
"Elapsed: $(elapsed_str) | Total: $(credit_used)

$(eval_summary)" "high" "checkered_flag"

log "Done."
