#!/bin/bash
# control_msf_v2.sh — MSF v2 experiment control panel (3‑seed, 3‑domain chain)
# Cherry‑picks best features from existing .sh scripts.

# Usage:
#   bash control_msf_v2.sh <ntfy_topic>
# Optional env vars:
#   SEEDS=3                # how many seeds (default 3)
#   STEPS=2000             # steps per domain
#   BS=16                  # batch size
#   RECORD_GRAD_NORMS=1    # enable gradient norm logging on first seed first domain
#   RUN_FULL_FT=1          # also run full fine‑tuning chain (same seeds)
#   RUN_MSF_V1=1           # also run MSF v1 chain (uniform mask) with train.py

set -eo pipefail

NTFY_TOPIC="${1:-proteus-aman-2026}"
SEEDS="${SEEDS:-3}"
STEPS="${STEPS:-2000}"
BS="${BS:-16}"
N_EVAL=100                         # eval samples per domain
RECORD_GRAD_NORMS="${RECORD_GRAD_NORMS:-0}"   # 1 = yes
RUN_FULL_FT="${RUN_FULL_FT:-0}"
RUN_MSF_V1="${RUN_MSF_V1:-0}"

# Budget tracking
START_TIME=$(date +%s)
RATE=1.99   # $/hour

LOG="results/run_msf_v2_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="results/.live_status_msf_v2.json"
STEP_FILE="results/.current_step_msf_v2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"
export PYTORCH_TUNABLEOP_ENABLED=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1

mkdir -p results /scratch/checkpoints

# ─────────────────────────────────────────────
# Helper functions (from existing scripts)
# ─────────────────────────────────────────────
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
    local need_gb=30   # safe margin
    log "Disk check: /scratch ${avail_gb}GB free (need ${need_gb}GB)"
    if (( avail_gb < need_gb )); then
        log "ABORT: low disk space."
        notify "MSF v2 ABORTED" "Disk full: ${avail_gb}GB on /scratch" "urgent" "rotating_light"
        exit 1
    fi
}

# ── Safe cleanup of stale processes and VRAM ──
cleanup_gpu() {
    log "Killing stale Python processes..."
    for pid in $(pgrep -f "python train" 2>/dev/null); do
        log "  Killing PID $pid"; kill -9 "$pid" 2>/dev/null || true
    done
    for pid in $(pgrep -f "python eval.py" 2>/dev/null); do
        log "  Killing PID $pid"; kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
    # Force PyTorch cache flush
    python3 -c "import torch, gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()" 2>&1 | tee -a "$LOG"
    log "VRAM state after cleanup:"
    rocm-smi --showmeminfo vram 2>&1 | tee -a "$LOG" || true
}

# ── Training wrappers ──
run_train_msf_v2() {
    local domain="$1" out_dir="$2" seed="$3"
    shift 3
    check_disk
    set_step "train_msf_v2/$domain/seed${seed}"
    log "START $CURRENT_STEP -> $out_dir"
    start_status_renderer
    python train2.py --domain "$domain" --out_dir "$out_dir" \
        --max_steps "$STEPS" --batch_size "$BS" --seed "$seed" \
        --status_file "$STATUS_FILE" \
        --record_grad_norms \
        "$@" 2>&1 | tee -a "$LOG"
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

run_train_full_ft() {
    local domain="$1" out_dir="$2" seed="$3"
    shift 3
    check_disk
    set_step "train_full/$domain/seed${seed}"
    log "START $CURRENT_STEP -> $out_dir"
    start_status_renderer
    python train2.py --domain "$domain" --out_dir "$out_dir" \
        --max_steps "$STEPS" --batch_size "$BS" --seed "$seed" \
        --status_file "$STATUS_FILE" \
        --full_ft \
        "$@" 2>&1 | tee -a "$LOG"
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
        print(f"{e['label']}: med={d.get('medical','?')} leg={d.get('legal','?')} cod={d.get('code','?')}")
EOF
}

# ─────────────────────────────────────────────
# Start of world
# ─────────────────────────────────────────────
trap 'on_error $LINENO' ERR

on_error() {
    local exit_code=$? line=$1 step
    step=$(get_step)
    stop_status_renderer
    log "CRASH at line $line (exit $exit_code) during: $step"
    notify "MSF v2 CRASHED" \
"Step: $step
Exit: $exit_code | Elapsed: $(elapsed_str) | Spent: $(credit_used)
Last: $(last_clean_log)" "urgent" "rotating_light"
    kill "${HEARTBEAT_PID:-}" 2>/dev/null || true
    exit $exit_code
}

heartbeat() {
    while true; do
        sleep 1800
        notify "MSF v2 heartbeat" \
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
log "MSF v2 control panel starting"
log "Seeds: $SEEDS | Steps/domain: $STEPS | Batch: $BS | Eval: $N_EVAL"
log "Record grad norms: $RECORD_GRAD_NORMS"
log "Full FT chain: $RUN_FULL_FT"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="

notify "MSF v2 experiment started" \
"Seeds: $SEEDS | Steps: $STEPS | BS: $BS | Topic: $NTFY_TOPIC" "default" "rocket"

# ══════════════════════════════════════════════
# MSF v2 chains (3 seeds, Medical → Legal → Code)
# ══════════════════════════════════════════════
log "=== MSF v2 chains ==="

for seed in $(seq 1 $SEEDS); do
    log "--- Seed $seed / $SEEDS ---"
    notify "MSF v2 seed $seed started" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

    MED_DIR="/scratch/checkpoints/msf_v2_seed${seed}/medical"
    LEG_DIR="/scratch/checkpoints/msf_v2_seed${seed}/legal"
    COD_DIR="/scratch/checkpoints/msf_v2_seed${seed}/code"

    # Optionally record grad norms only on seed 1 and first domain
    GRAD_NORM_FLAG=""
    if [[ "$RECORD_GRAD_NORMS" == "1" && $seed -eq 1 ]]; then
        GRAD_NORM_FLAG="--record_grad_norms"
    fi

    # Medical
    run_train_msf_v2 medical "$MED_DIR" "$seed" $GRAD_NORM_FLAG
    run_eval "$MED_DIR" "msf_v2_seed${seed}_after_medical"

    # Legal (start from medical checkpoint)
    run_train_msf_v2 legal "$LEG_DIR" "$seed" --start_from "$MED_DIR"
    run_eval "$LEG_DIR" "msf_v2_seed${seed}_after_legal"

    # Code (start from legal checkpoint)
    run_train_msf_v2 code "$COD_DIR" "$seed" --start_from "$LEG_DIR"
    run_eval "$COD_DIR" "msf_v2_seed${seed}_after_code"

    log "Seed $seed complete"
    notify "MSF v2 seed $seed done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "high" "white_check_mark"
done

# ══════════════════════════════════════════════
# Optional Full FT baseline (same seeds)
# ══════════════════════════════════════════════
if [[ "$RUN_FULL_FT" == "1" ]]; then
    log "=== Full FT baseline chains (using train2.py --full_ft) ==="
    for seed in $(seq 1 $SEEDS); do
        log "--- Full FT seed $seed ---"
        notify "Full FT seed $seed started" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

        FT_MED="/scratch/checkpoints/full_seed${seed}/medical"
        FT_LEG="/scratch/checkpoints/full_seed${seed}/legal"
        FT_COD="/scratch/checkpoints/full_seed${seed}/code"

        run_train_full_ft medical "$FT_MED" "$seed"
        run_eval "$FT_MED" "full_seed${seed}_after_medical"

        run_train_full_ft legal "$FT_LEG" "$seed" --start_from "$FT_MED"
        run_eval "$FT_LEG" "full_seed${seed}_after_legal"

        run_train_full_ft code "$FT_COD" "$seed" --start_from "$FT_LEG"
        run_eval "$FT_COD" "full_seed${seed}_after_code"

        notify "Full FT seed $seed done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "high" "white_check_mark"
    done
fi

# ══════════════════════════════════════════════
# Optional MSF v1 baseline (uniform mask from train.py)
# ══════════════════════════════════════════════
if [[ "$RUN_MSF_V1" == "1" ]]; then
    log "=== MSF v1 baseline chains ==="
    for seed in $(seq 1 $SEEDS); do
        log "--- MSF v1 seed $seed ---"
        notify "MSF v1 seed $seed started" "Elapsed: $(elapsed_str) | Spent: $(credit_used)" "low"

        V1_MED="/scratch/checkpoints/msf_v1_seed${seed}/medical"
        V1_LEG="/scratch/checkpoints/msf_v1_seed${seed}/legal"
        V1_COD="/scratch/checkpoints/msf_v1_seed${seed}/code"

        run_train_full medical "$V1_MED" --condition proteus   # reuse run_train_full but with condition proteus
        run_eval "$V1_MED" "msf_v1_seed${seed}_after_medical"

        run_train_full legal "$V1_LEG" --start_from "$V1_MED" --condition proteus
        run_eval "$V1_LEG" "msf_v1_seed${seed}_after_legal"

        run_train_full code "$V1_COD" --start_from "$V1_LEG" --condition proteus
        run_eval "$V1_COD" "msf_v1_seed${seed}_after_code"

        notify "MSF v1 seed $seed done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" "high" "white_check_mark"
    done
fi

# ─────────────────────────────────────────────
# Final wrap‑up
# ─────────────────────────────────────────────
log "=============================="
log "All experiments completed."
log "Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "MSF v2 ALL DONE" \
"Seeds: $SEEDS | Steps: $STEPS
Elapsed: $(elapsed_str) | Total: $(credit_used)

$(eval_summary)" "high" "checkered_flag"

log "Done."