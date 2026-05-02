#!/bin/bash
# control_baselines.sh — EWC + Replay baselines (seed 1 only)
# MSF v2 and Full FT already done. This just adds the expensive baselines.
#
# Usage:
#   bash control_baselines.sh <ntfy_topic>
#
# ETA: ~3–4 hours

set -eo pipefail

NTFY_TOPIC="${1:-proteus-aman-2026}"
SEED=1                    # single seed, enough for the comparison
STEPS=2000
BS=16
N_EVAL=100

START_TIME=$(date +%s)
RATE=1.99

LOG="results/run_baselines_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="results/.live_status_baselines.json"
STEP_FILE="results/.current_step_baselines"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"
export PYTORCH_TUNABLEOP_ENABLED=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1

mkdir -p results /scratch/checkpoints

# ── Helpers ──
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

set_step() { echo "$1" > "$STEP_FILE"; }
get_step()  { cat "$STEP_FILE" 2>/dev/null || echo "unknown"; }

last_clean_log() {
    grep -E "^\[20|loss|perplexity|DONE|START|CRASH" "$LOG" 2>/dev/null | tail -1 || echo "no log yet"
}

format_status_line() {
    python3 - "$STATUS_FILE" <<'PY'
import json, math, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except:
    print(""); raise SystemExit(0)
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

check_disk() {
    local avail_kb=$(df /scratch --output=avail 2>/dev/null | tail -1)
    local avail_gb=$(( avail_kb / 1024 / 1024 ))
    local need_gb=30
    log "Disk: /scratch ${avail_gb}GB free (need ${need_gb}GB)"
    if (( avail_gb < need_gb )); then
        log "ABORT: low disk space."
        notify "Baselines ABORTED" "Disk full" "urgent" "rotating_light"
        exit 1
    fi
}

cleanup_gpu() {
    log "Killing stale Python processes..."
    for pid in $(pgrep -f "python train" 2>/dev/null); do
        log "  Killing PID $pid"; kill -9 "$pid" 2>/dev/null || true
    done
    for pid in $(pgrep -f "python eval.py" 2>/dev/null); do
        log "  Killing PID $pid"; kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
    python3 -c "import torch, gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()" 2>&1 | tee -a "$LOG"
    log "VRAM state after cleanup:"
    rocm-smi --showmeminfo vram 2>&1 | tee -a "$LOG" || true
}

# ── Wrappers ──
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

run_train_ewc() {
    local domain="$1" out_dir="$2" ewc_state_path="$3"
    shift 3
    cleanup_gpu          # <-- it OOM'd without this
    check_disk
    set_step "train_ewc/$domain"
    log "START $CURRENT_STEP -> $out_dir (ewc_state=${ewc_state_path:-none})"
    start_status_renderer
    local ewc_arg=""
    [[ -n "$ewc_state_path" && "$ewc_state_path" != "none" ]] && ewc_arg="--ewc_state $ewc_state_path"
    python train.py --domain "$domain" --condition ewc --out_dir "$out_dir" \
        --max_steps "$STEPS" --batch_size "$BS" --grad_accum 1 \
        --status_file "$STATUS_FILE" \
        $ewc_arg \
        "$@" 2>&1 | tee -a "$LOG"
    stop_status_renderer
    log "DONE  $CURRENT_STEP"
}

run_train_replay() {
    local domain="$1" out_dir="$2" replay_buffer_path="$3"
    shift 3
    cleanup_gpu          # <-- it OOM'd without this
    check_disk
    set_step "train_replay/$domain"
    log "START $CURRENT_STEP -> $out_dir (replay_buffer=${replay_buffer_path:-none})"
    start_status_renderer
    local replay_arg=""
    [[ -n "$replay_buffer_path" && "$replay_buffer_path" != "none" ]] && replay_arg="--replay_buffer $replay_buffer_path"
    python train.py --domain "$domain" --condition replay --out_dir "$out_dir" \
        --max_steps "$STEPS" --batch_size "$BS" --grad_accum 1 \
        --status_file "$STATUS_FILE" \
        $replay_arg \
        "$@" 2>&1 | tee -a "$LOG"
    stop_status_renderer
    log "DONE  $CURRENT_STEP"
}
build_replay_buffer() {
    local domain="$1" out_path="$2" n_samples="${3:-5000}"
    log "Building replay buffer: $domain -> $out_path ($n_samples samples)"
    python3 -c "
import json, random
with open('data/$domain/train.jsonl') as f:
    data = f.readlines()
random.shuffle(data)
with open('$out_path', 'w') as f:
    for line in data[:$n_samples]:
        f.write(line)
print(f'Saved {min(len(data), $n_samples)} lines')
"
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
        print(f"{e['label']}: med={d.get('medical','?')} leg={d.get('legal','?')} cod={d.get('code','?')}")
EOF
}

# ── Trap & heartbeat ──
trap 'on_error $LINENO' ERR
on_error() {
    local exit_code=$? line=$1 step
    step=$(get_step)
    stop_status_renderer
    log "CRASH at line $line (exit $exit_code) during: $step"
    notify "Baselines CRASHED" "Step: $step | Exit: $exit_code" "urgent" "rotating_light"
    kill "${HEARTBEAT_PID:-}" 2>/dev/null || true
    exit $exit_code
}

heartbeat() {
    while true; do
        sleep 1800
        notify "Baselines heartbeat" \
"Step: $(get_step)
Elapsed: $(elapsed_str) | Spent: $(credit_used)
Last: $(last_clean_log)" "low"
    done
}

heartbeat &
HEARTBEAT_PID=$!
trap 'stop_status_renderer; kill ${HEARTBEAT_PID:-} 2>/dev/null' EXIT

cleanup_gpu

log "=============================="
log "EWC + Replay baselines (seed 1)"
log "Steps: $STEPS | Batch: $BS"
log "ETA ~3-4 hours"
log "=============================="
notify "Baselines started (EWC & Replay, seed 1)" "ETA: ~3-4h" "default" "rocket"

# ── EWC ──
log "=== EWC (seed 1) ==="
EWC_MED="/scratch/checkpoints/ewc_seed1/medical"
EWC_LEG="/scratch/checkpoints/ewc_seed1/legal"
EWC_COD="/scratch/checkpoints/ewc_seed1/code"

run_train_ewc medical "$EWC_MED" "none"
run_eval "$EWC_MED" "ewc_seed1_after_medical"

run_train_ewc legal "$EWC_LEG" "$EWC_MED/fisher.pt" --start_from "$EWC_MED"
run_eval "$EWC_LEG" "ewc_seed1_after_legal"

run_train_ewc code "$EWC_COD" "$EWC_LEG/fisher.pt" --start_from "$EWC_LEG"
run_eval "$EWC_COD" "ewc_seed1_after_code"

log "EWC done"
notify "EWC done" "$(eval_summary)" "high" "white_check_mark"

# ── Replay ──
log "=== Replay (seed 1) ==="
REP_MED="/scratch/checkpoints/replay_seed1/medical"
REP_LEG="/scratch/checkpoints/replay_seed1/legal"
REP_COD="/scratch/checkpoints/replay_seed1/code"
REPLAY_DIR="/scratch/replay_buffers"
mkdir -p "$REPLAY_DIR"

MED_BUFFER="$REPLAY_DIR/medical_buffer.jsonl"
build_replay_buffer medical "$MED_BUFFER" 5000

run_train_replay medical "$REP_MED" "none"
run_eval "$REP_MED" "replay_seed1_after_medical"

run_train_replay legal "$REP_LEG" "$MED_BUFFER" --start_from "$REP_MED"
run_eval "$REP_LEG" "replay_seed1_after_legal"

LEG_BUFFER="$REPLAY_DIR/legal_buffer.jsonl"
build_replay_buffer legal "$LEG_BUFFER" 5000

run_train_replay code "$REP_COD" "$LEG_BUFFER" --start_from "$REP_LEG"
run_eval "$REP_COD" "replay_seed1_after_code"

log "Replay done"
notify "Replay done" "$(eval_summary)" "high" "white_check_mark"

log "=============================="
log "All baselines complete."
log "Elapsed: $(elapsed_str) | Spent: $(credit_used)"
log "=============================="
notify "Baselines ALL DONE" \
"EWC + Replay (seed 1)
Elapsed: $(elapsed_str) | Spent: $(credit_used)

$(eval_summary)" "high" "checkered_flag"

log "Done."