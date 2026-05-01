#!/usr/bin/env python3
"""
train_msf_v2.py
Matryoshka Subspace Freezing V2 — per‑layer adaptive core freeze.

Replaces the uniform mask of MSF v1 with a layer‑wise frozen rectangle
whose size is scaled by s_ℓ², where s_ℓ is the measured core–shell
coupling score (cosine similarity of gate_proj core outputs at full
width vs core‑only width).  Lone layers (s_ℓ ≥ 0.8) freeze heavily;
Tandem layers (s_ℓ ≤ 0.69) freeze only a tiny core.

Usage:
  python train_msf_v2.py --domain medical --out_dir checkpoints/msf_v2/med
  python train_msf_v2.py --domain legal --start_from checkpoints/msf_v2/med
  python train_msf_v2.py --domain code --start_from checkpoints/msf_v2/leg
"""

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ── Constants matching Gemma‑4‑E4B ──
MODEL_ID       = "google/gemma-4-E4B-it"
CORE_HIDDEN    = 1536
CORE_MID       = 6144
MAX_LENGTH     = 512

# ── Coupling scores from core_coupling_analysis.py ──
# Hard‑coded for resilience; a CSV can override via --coupling_csv.
DEFAULT_COUPLING = {
    0: 0.891, 1: 0.863, 2: 0.852, 3: 0.793, 4: 0.793,
    5: 0.840, 6: 0.855, 7: 0.789, 8: 0.758, 9: 0.691,
    10:0.660,11:0.684,12:0.773,13:0.754,14:0.555,15:0.609,
    16:0.535,17:0.570,18:0.613,19:0.625,20:0.727,21:0.719,
    22:0.801,23:0.711,24:0.633,25:0.531,26:0.609,27:0.617,
    28:0.668,29:0.742,30:0.719,31:0.645,32:0.770,33:0.758,
    34:0.770,35:0.648,36:0.613,37:0.555,38:0.582,39:0.645,
    40:0.711,41:0.695,
}


def load_coupling(csv_path=None):
    """Return dict layer_idx -> s_ℓ (float)."""
    if csv_path:
        import csv
        scores = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer"]) if "layer" in row else int(row["Layer"])
                score = float(row["coupling"]) if "coupling" in row else float(row["s_l"])
                scores[layer] = score
        return scores
    else:
        return DEFAULT_COUPLING


def per_layer_freeze_dims(coupling):
    """
    For each layer, compute frozen rectangle:
        m_ℓ (intermediate dim) = round(s_ℓ² * CORE_MID)
        n_ℓ (hidden dim)       = round(s_ℓ² * CORE_HIDDEN)
    """
    dims = {}
    for layer, s in coupling.items():
        f = s ** 2
        m = round(f * CORE_MID)
        n = round(f * CORE_HIDDEN)
        dims[layer] = (m, n)
    return dims


class GradientNormHook:
    """
    Records Frobenius norm of core and shell gradients at the first
    backward pass after being reset.  Used to log gradient flow at
    the start of each domain.
    """
    def __init__(self, layer_idx, proj_type, dims_dict):
        self.layer_idx = layer_idx
        self.proj_type = proj_type  # "gate", "up", or "down"
        self.dims = dims_dict       # (m, n) for this layer
        self.core_norm = None
        self.shell_norm = None
        self._done = False

    def hook(self, grad):
        if self._done:
            return grad   # only measure first backward
        m, n = self.dims
        with torch.no_grad():
            # copy to avoid disturbing autograd (grad is a leaf tensor, but we can compute norm)
            grad_copy = grad.detach()
            if self.proj_type != "down":
                core = grad_copy[:m, :n]
                shell = torch.cat([
                    grad_copy[m:, :].flatten(),
                    grad_copy[:m, n:].flatten(),
                ])
            else:
                core = grad_copy[:n, :m]
                shell = torch.cat([
                    grad_copy[n:, :].flatten(),
                    grad_copy[:n, m:].flatten(),
                ])
            self.core_norm = core.norm().item()
            self.shell_norm = shell.norm().item()
        self._done = True
        return grad   # no modification (MSF v2 mask is separate)


def register_v2_hooks(model, coupling, record_grad_norms=False):
    """
    Register two types of hooks on every FFN weight:
      1. Gradient mask (zero the frozen core rectangle).
      2. (Optional) Gradient norm measurement at first backward.
    Returns tuple (list of all hooks, dict of norm hooks keyed by (layer,proj)).
    """
    layers = model.model.language_model.layers
    dims = per_layer_freeze_dims(coupling)
    all_hooks = []
    norm_hooks = {}

    for i, layer in enumerate(layers):
        if i not in dims:
            # fallback: freeze full core (safe, v1 behavior)
            m, n = CORE_MID, CORE_HIDDEN
        else:
            m, n = dims[i]

        # Mask hook
        def make_mask_hook(proj, _m, _n):
            def hook(grad):
                if proj in ("gate", "up"):
                    grad[:_m, :_n] = 0.0
                else:
                    grad[:_n, :_m] = 0.0
                return grad
            return hook

        all_hooks.append(layer.mlp.gate_proj.weight.register_hook(
            make_mask_hook("gate", m, n)))
        all_hooks.append(layer.mlp.up_proj.weight.register_hook(
            make_mask_hook("up", m, n)))
        all_hooks.append(layer.mlp.down_proj.weight.register_hook(
            make_mask_hook("down", m, n)))

        # Gradient norm measurement hooks (optional)
        if record_grad_norms:
            for proj_name in ("gate", "up", "down"):
                proj = getattr(layer.mlp, f"{proj_name}_proj")
                ngh = GradientNormHook(i, proj_name, (m, n) if proj_name != "down" else (n, m))
                proj.weight.register_hook(ngh.hook)
                norm_hooks[(i, proj_name)] = ngh

    print(f"[msf_v2] Registered {len(all_hooks)} gradient mask hooks across {len(layers)} layers.")
    if record_grad_norms:
        print(f"[msf_v2] Gradient norm hooks active for first backward pass.")
    return all_hooks, norm_hooks


class ProgressWriter:
    """Writes progress.json after each domain."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = []

    def record(self, entry: dict):
        self.data.append(entry)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


class StatusWriter:
    """Minimal status file emitter (same as train.py)."""
    def __init__(self, path):
        self.path = Path(path) if path else None

    def emit(self, payload):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = dict(payload)
            data["ts"] = time.time()
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except Exception:
            pass


class TrainStatusCallback(TrainerCallback):
    """Streams step‑level metrics for shell status renderer (like original)."""
    def __init__(self, writer, domain, total_steps):
        self.writer = writer
        self.domain = domain
        self.total_steps = total_steps
        self.start_time = None
        self.last_loss = None

    def _emit(self, state, step):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = max(time.time() - self.start_time, 1e-6)
        it_s = step / elapsed if step > 0 else 0.0
        eta_s = None
        if self.total_steps and step < self.total_steps and it_s > 0:
            eta_s = (self.total_steps - step) / it_s

        payload = {
            "phase": "train",
            "state": state,
            "condition": "msf_v2",
            "domain": self.domain,
            "step": step,
            "total_steps": self.total_steps,
            "it_s": it_s,
            "eta_s": eta_s,
        }
        if self.last_loss is not None:
            payload["loss"] = self.last_loss
        self.writer.emit(payload)

    def on_train_begin(self, args, state, control, **kwargs):
        self._emit("running", state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        self._emit("running", state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if isinstance(logs, dict) and logs.get("loss") is not None:
            self.last_loss = logs["loss"]
        self._emit("running", state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self._emit("done", state.global_step)


# ─────────────────────────────────────────────
# Dataset helpers (mirrors train.py)
# ─────────────────────────────────────────────
DATA_DIR = Path("data")
RESPONSE_KEY = "### Response:\n"


def load_domain(domain: str, split: str = "train") -> Dataset:
    path = DATA_DIR / domain / f"{split}.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_prompt_parts(row):
    instruction = row["instruction"]
    inp = row.get("input", "").strip()
    output = row["output"]
    if inp:
        prefix = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n{RESPONSE_KEY}"
    else:
        prefix = f"### Instruction:\n{instruction}\n\n{RESPONSE_KEY}"
    return prefix, output


def tokenize_dataset(raw_ds, tokenizer):
    def tokenize_fn(batch):
        input_ids_list, labels_list = [], []
        for i in range(len(batch["instruction"])):
            prefix, response = format_prompt_parts({
                "instruction": batch["instruction"][i],
                "input": batch.get("input", [""] * len(batch["instruction"]))[i],
                "output": batch["output"][i],
            })
            prefix_ids = tokenizer(prefix, truncation=False, add_special_tokens=True)["input_ids"]
            full_ids = tokenizer(prefix + response, truncation=True, max_length=MAX_LENGTH,
                                 add_special_tokens=True)["input_ids"]
            n_prefix = min(len(prefix_ids), len(full_ids) - 1)
            labels = [-100] * n_prefix + full_ids[n_prefix:]
            input_ids_list.append(full_ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}
    return raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["medical", "legal", "code"])
    parser.add_argument("--start_from", default=None, help="Path to previous checkpoint")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--coupling_csv", default=None, help="Optional CSV with layer,coupling")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--status_file", default=None, help="JSON file for live status rendering")
    parser.add_argument("--progress_file", default="progress.json", help="Per‑domain checkpoint log")
    parser.add_argument("--record_grad_norms", action="store_true",
                        help="Record core/shell gradient norms at first step")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_path = args.start_from if args.start_from else MODEL_ID
    print(f"Loading model from {load_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    # Strip vision/audio as in original pipeline
    for attr in ["vision_tower", "audio_encoder", "multi_modal_projector"]:
        if hasattr(model, attr):
            delattr(model, attr)
    torch.cuda.empty_cache()

    # ── Load coupling scores ──
    coupling = load_coupling(args.coupling_csv)

    # ── Register hooks (MSF v2 mask + optional grad norm recording) ──
    hooks, norm_hooks = register_v2_hooks(model, coupling, record_grad_norms=args.record_grad_norms)
    # Attention remains fully trainable (no nested structure)
    print("[msf_v2] Attention layers fully trainable.")

    # ── Dataset ──
    raw_ds = load_domain(args.domain, split="train")
    max_samples = args.max_steps * args.batch_size * 2  # safety margin
    if len(raw_ds) > max_samples:
        raw_ds = raw_ds.select(range(max_samples))
    tokenized = tokenize_dataset(raw_ds, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=128)

    # ── Training ──
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=25,
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    # Status and progress logging
    status = StatusWriter(args.status_file)
    progress = ProgressWriter(out_dir / args.progress_file)
    trainer.add_callback(TrainStatusCallback(status, args.domain, args.max_steps))

    print(f"[msf_v2] Starting training for {args.max_steps} steps...")
    trainer.train()

    # ── Save final checkpoint ──
    for h in hooks:
        h.remove()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save progress entry
    progress.record({
        "domain": args.domain,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "save_path": str(out_dir),
    })

    # Log gradient norms if recorded
    if args.record_grad_norms and norm_hooks:
        norm_rows = []
        for (layer, proj), nh in sorted(norm_hooks.items()):
            if nh.core_norm is not None:
                norm_rows.append(f"  L{layer:2d} {proj:4s} core={nh.core_norm:.4f} shell={nh.shell_norm:.4f}")
        if norm_rows:
            print("\n[msf_v2] First‑step gradient norms:")
            print("\n".join(norm_rows))

    print(f"\n[msf_v2] Done. Model saved to {out_dir}")


if __name__ == "__main__":
    main()