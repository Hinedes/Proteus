#!/bin/bash
chmod +x "$0"
# Proteus — 4-Domain Sequential Chain Runner
# Runs Proteus vs Full fine-tune across medical → legal → code → multilingual.
# Single-domain baselines (LoRA, EWC, Replay) are already complete.
#
# Usage:
#   ./run_all.sh YOUR_NTFY_TOPIC

NTFY_TOPIC="${1:-proteus-aman-2026}"
LOG="results/run_all.log"
STEPS=2000
N_EVAL=100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"
START_TIME=$(date +%s)
RATE=1.99   # USD/hr for MI300X

mkdir -p results

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

# ─────────────────────────────────────────────
# Failure handler
# ─────────────────────────────────────────────
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
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("", end="")
    raise SystemExit(0)

def fmt_eta(seconds):
    if seconds is None:
        return "--:--"
    try:
        secs = int(max(0, float(seconds)))
    except Exception:
        return "--:--"
    mins, sec = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs:02d}:{mins:02d}:{sec:02d}"
    return f"{mins:02d}:{sec:02d}"

phase = data.get("phase")
state = str(data.get("state", ""))

if phase == "train":
    step = int(data.get("step", 0) or 0)
    total = int(data.get("total_steps", 0) or 0)
    pct = (100.0 * step / total) if total > 0 else 0.0
    it_s = float(data.get("it_s", 0.0) or 0.0)
    eta = fmt_eta(data.get("eta_s"))
    loss = data.get("loss")
    if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
        loss_txt = f"{float(loss):.4f}"
    else:
        loss_txt = "--"
    condition = str(data.get("condition", "?"))
    domain = str(data.get("domain", "?"))
    print(
        f"[train {condition}/{domain}] {step}/{total} ({pct:5.1f}%) "
        f"| {it_s:6.2f} it/s | eta {eta} | loss {loss_txt} | {state}",
        end="",
    )
elif phase == "eval":
    domain = str(data.get("domain", "?"))
    domain_index = int(data.get("domain_index", 0) or 0)
    domains_total = int(data.get("domains_total", 0) or 0)
    sample = int(data.get("sample", 0) or 0)
    sample_total = int(data.get("sample_total", 0) or 0)
    overall_step = int(data.get("overall_step", 0) or 0)
    overall_total = int(data.get("overall_total", 0) or 0)
    pct = (100.0 * overall_step / overall_total) if overall_total > 0 else 0.0
    it_s = float(data.get("it_s", 0.0) or 0.0)
    ppl = data.get("ppl_so_far")
    if isinstance(ppl, (int, float)) and math.isfinite(float(ppl)):
        ppl_txt = f"{float(ppl):.3f}"
    else:
        ppl_txt = "--"
    print(
        f"[eval {domain_index}/{domains_total} {domain}] sample {sample}/{sample_total} "
        f"| overall {overall_step}/{overall_total} ({pct:5.1f}%) "
        f"| {it_s:6.2f} it/s | ppl~ {ppl_txt} | {state}",
        end="",
    )
else:
    print("", end="")
PY
}

start_status_renderer() {
    if [[ ! -t 2 ]]; then
        return
    fi
    rm -f "$STATUS_FILE"
    : > "$STATUS_FILE"
    (
        while true; do
            if [[ -s "$STATUS_FILE" ]]; then
                line=$(format_status_line)
                if [[ -n "$line" ]]; then
                    printf "\r\033[2K%s" "$line" >&2
                fi
            fi
            sleep 1
        done
    ) &
    STATUS_RENDER_PID=$!
}

stop_status_renderer() {
    if [[ -n "${STATUS_RENDER_PID:-}" ]]; then
        kill "$STATUS_RENDER_PID" 2>/dev/null || true
        wait "$STATUS_RENDER_PID" 2>/dev/null || true
        STATUS_RENDER_PID=""
    fi
    if [[ -t 2 ]]; then
        printf "\r\033[2K" >&2
    fi
    rm -f "$STATUS_FILE"
}

on_error() {
    local exit_code=$?
    local line=$1
    local step
    step=$(get_step)
    stop_status_renderer
    log "CRASH at line $line (exit $exit_code) during: $step"
    local tail_log
    tail_log=$(last_clean_log)
    notify "Proteus CRASHED" \
"Step: $step
Exit: $exit_code
Elapsed: $(elapsed_str) | Spent: $(credit_used)

Last: $tail_log

Resume manually from failed step." \
        "urgent" "rotating_light"
    kill "${HEARTBEAT_PID:-}" 2>/dev/null || true
    exit $exit_code
}

trap 'on_error $LINENO' ERR

# ─────────────────────────────────────────────
# Heartbeat (every 30 min)
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
trap 'on_error $LINENO' ERR
trap 'stop_status_renderer; kill ${HEARTBEAT_PID:-} 2>/dev/null' EXIT

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
run_train() {
    local domain="$1"
    local condition="$2"
    shift 2
    set_step "train/$condition/$domain"
    log "START $CURRENT_STEP $*"
    start_status_renderer
    python train.py --domain "$domain" --condition "$condition" \
        --max_steps "$STEPS" --batch_size 16 --grad_accum 1 --compile --status_file "$STATUS_FILE" "$@" 2>&1 | tee -a "$LOG"
    stop_status_renderer
    log "DONE  $CURRENT_STEP"
}

run_eval() {
    local checkpoint="$1"
    local label="$2"
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
        med = d.get("medical", "?")
        leg = d.get("legal", "?")
        cod = d.get("code", "?")
        mul = d.get("multilingual", "?")
        print(f"{e['label']}: med={med} leg={leg} cod={cod} mul={mul}")
EOF
}

# ─────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────
log "=============================="
log "Proteus 4-domain sequential chains"
log "ntfy topic: $NTFY_TOPIC"
log "=============================="

notify "Proteus 4-domain chains started" \
"Chains: Proteus, Full
4 domains × 500 steps × 2 conditions = 4000 steps total
Topic: $NTFY_TOPIC" \
    "default" "rocket"

# ─────────────────────────────────────────────
# 4-Domain Sequential Chain — Proteus
# ─────────────────────────────────────────────
log "--- CHAIN: proteus (medical → legal → code → multilingual) ---"

run_train medical proteus
run_eval  checkpoints/proteus/medical proteus_seq_after_medical

run_train legal   proteus --start_from checkpoints/proteus/medical
run_eval  checkpoints/proteus/legal   proteus_seq_after_legal

run_train code    proteus --start_from checkpoints/proteus/legal
run_eval  checkpoints/proteus/code    proteus_seq_after_code

run_train multilingual proteus --start_from checkpoints/proteus/code
run_eval  checkpoints/proteus/multilingual proteus_seq_after_multilingual

notify "Proteus chain done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" \
    "default" "white_check_mark"

# ─────────────────────────────────────────────
# 4-Domain Sequential Chain — Full
# ─────────────────────────────────────────────
log "--- CHAIN: full (medical → legal → code → multilingual) ---"

run_train medical full
run_eval  checkpoints/full/medical full_seq_after_medical

run_train legal   full --start_from checkpoints/full/medical
run_eval  checkpoints/full/legal   full_seq_after_legal

run_train code    full --start_from checkpoints/full/legal
run_eval  checkpoints/full/code    full_seq_after_code

run_train multilingual full --start_from checkpoints/full/code
run_eval  checkpoints/full/multilingual full_seq_after_multilingual

notify "Full chain done" \
"$(eval_summary)
Elapsed: $(elapsed_str) | Spent: $(credit_used)" \
    "default" "white_check_mark"

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
log "=============================="
log "All chains complete."
log "Elapsed: $(elapsed_str) | Total spend: $(credit_used)"
log "=============================="

notify "Proteus COMPLETE" \
"Both 4-domain chains done!
Elapsed: $(elapsed_str) | Total: $(credit_used)

$(eval_summary)" \
    "high" "checkered_flag"

log "Done."
