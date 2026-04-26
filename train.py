"""
Proteus — Fine-tune Wrapper
Trains Gemma 4 E4B on one domain under a specified condition.

Conditions:
  proteus   — Core frozen via gradient hook (the thesis)
  full      — Full fine-tune, no protection
  lora      — LoRA adapters on attention (q_proj, v_proj) — standard baseline
  lora_ffn  — LoRA adapters on FFN (gate/up/down proj) — same location as Proteus
  ewc       — Elastic Weight Consolidation (importance-weighted regularization)
  replay    — Data replay from previous domains

Usage:
  python train.py --domain medical --condition proteus
  python train.py --domain legal   --condition full   --max_steps 200
  python train.py --domain code    --condition lora
  python train.py --domain code    --condition lora_ffn
  python train.py --domain code    --condition proteus --compile

  # EWC: first domain needs no prior state; subsequent domains load it
  python train.py --domain medical --condition ewc
  python train.py --domain legal   --condition ewc --ewc_state checkpoints/ewc/medical/fisher.pt

  # Replay: pass a .jsonl file of buffered prior-domain examples
  python train.py --domain legal   --condition replay --replay_buffer data/replay_buffer.jsonl
"""

import argparse
import atexit
import gc
import json
import logging
import multiprocessing as mp
import os
import random
import time
import warnings
from pathlib import Path

# Silence noisy third-party warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformer_engine").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

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

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID    = "google/gemma-4-E4B-it"
DATA_DIR    = Path("data")
CKPT_DIR    = Path("checkpoints")
CORE_HIDDEN = 1536
CORE_MID    = 6144
MAX_LENGTH  = 512


def validate_args(args):
    if not (0.0 <= args.replay_ratio < 1.0):
        raise ValueError("--replay_ratio must satisfy 0.0 <= replay_ratio < 1.0.")
    if args.ewc_samples <= 0:
        raise ValueError("--ewc_samples must be > 0.")


class StatusWriter:
    """Best-effort JSON status writer for external progress rendering."""

    def __init__(self, path):
        self.path = Path(path) if path else None

    def emit(self, payload: dict):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = dict(payload)
            data["ts"] = time.time()
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self.path)
        except Exception:
            pass


def cleanup_runtime():
    try:
        children = mp.active_children()
        for child in children:
            try:
                child.terminate()
            except Exception:
                pass
        for child in children:
            try:
                child.join(timeout=1)
            except Exception:
                pass
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


atexit.register(cleanup_runtime)


class TrainStatusCallback(TrainerCallback):
    """Streams step-level metrics for shell-side single-line rendering."""

    def __init__(self, writer: StatusWriter, condition: str, domain: str, total_steps: int):
        self.writer = writer
        self.condition = condition
        self.domain = domain
        self.total_steps = total_steps
        self._start_time = None
        self._last_loss = None

    def _emit(self, state_label: str, step: int):
        if self._start_time is None:
            self._start_time = time.time()

        total = max(int(self.total_steps), 0)
        step = max(int(step), 0)
        elapsed = max(time.time() - self._start_time, 1e-6)
        it_s = step / elapsed if step > 0 else 0.0
        eta_s = None
        if total > 0 and it_s > 0.0 and step < total:
            eta_s = (total - step) / it_s

        payload = {
            "phase": "train",
            "state": state_label,
            "condition": self.condition,
            "domain": self.domain,
            "step": step,
            "total_steps": total,
            "it_s": it_s,
            "eta_s": eta_s,
        }
        if self._last_loss is not None:
            payload["loss"] = float(self._last_loss)
        self.writer.emit(payload)

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
        self._emit("running", state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        self._emit("running", state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        log_data = logs if isinstance(logs, dict) else kwargs.get("logs")
        if isinstance(log_data, dict) and log_data.get("loss") is not None:
            self._last_loss = log_data["loss"]
        self._emit("running", state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self._emit("done", state.global_step)

# ─────────────────────────────────────────────
# aiter — MI300X fused kernel optimizations
# ─────────────────────────────────────────────
def apply_aiter_optimizations(model):
    """
    Monkey-patches Gemma4 MLP forward with aiter's fused silu_and_mul kernel.
    Replaces: silu(gate_proj(x)) * up_proj(x)  →  aiter.silu_and_mul (one kernel)
    Safe: wrapped in try/except, falls back to stock ops on any failure.
    """
    try:
        import aiter as _aiter

        layers = model.model.language_model.layers
        patched = 0

        def make_aiter_mlp_forward(original_mlp):
            def forward(hidden_states):
                gate = original_mlp.gate_proj(hidden_states)
                up   = original_mlp.up_proj(hidden_states)
                fused_input = torch.cat([gate, up], dim=-1)
                activated   = torch.empty_like(gate)
                _aiter.silu_and_mul(activated, fused_input)
                return original_mlp.down_proj(activated)
            return forward

        for layer in layers:
            mlp = layer.mlp
            mlp.forward = make_aiter_mlp_forward(mlp)
            patched += 1

        print(f"[aiter] Patched {patched} MLP layers with fused silu_and_mul.")
    except Exception as e:
        print(f"[aiter] Skipping optimizations: {e}")


# ─────────────────────────────────────────────
# Gradient hook (Proteus condition)
# ─────────────────────────────────────────────
def register_hooks(model):
    hooks = []
    layers = model.model.language_model.layers

    def make_hook(proj_type):
        def hook(grad):
            if proj_type in ("gate", "up"):
                grad[:CORE_MID, :CORE_HIDDEN] = 0.0
            else:
                grad[:CORE_HIDDEN, :CORE_MID] = 0.0
            return grad
        return hook

    for layer in layers:
        hooks.append(layer.mlp.gate_proj.weight.register_hook(make_hook("gate")))
        hooks.append(layer.mlp.up_proj.weight.register_hook(make_hook("up")))
        hooks.append(layer.mlp.down_proj.weight.register_hook(make_hook("down")))

    print(f"[proteus] Registered {len(hooks)} gradient hooks across {len(layers)} layers.")
    return hooks


# ─────────────────────────────────────────────
# EWC — Fisher matrix computation + custom Trainer
# ─────────────────────────────────────────────
def compute_fisher(model, dataset, n_samples=200, batch_size=16):
    model.eval()
    fisher     = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    opt_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    n_batches = 0

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(dataset[j]["input_ids"]) for j in batch_idx],
            batch_first=True, padding_value=0
        ).to(next(model.parameters()).device)
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(dataset[j]["labels"]) for j in batch_idx],
            batch_first=True, padding_value=-100
        ).to(next(model.parameters()).device)

        model.zero_grad()
        out  = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_batches += 1

    if n_batches == 0:
        model.train()
        raise RuntimeError(
            "[ewc] Fisher estimation produced zero batches. "
            "Ensure the dataset is non-empty and --ewc_samples > 0."
        )

    for n in fisher:
        fisher[n] /= n_batches

    model.train()
    return fisher, opt_params


def save_ewc_state(fisher, opt_params, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"fisher": fisher, "opt_params": opt_params}, path / "fisher.pt")
    print(f"[ewc] Fisher state saved to {path / 'fisher.pt'}")


def load_ewc_state(path: str, device=None):
    state = torch.load(path, map_location="cpu")
    fisher     = state["fisher"]
    opt_params = state["opt_params"]
    if device is not None:
        fisher     = {k: v.to(device, non_blocking=True) for k, v in fisher.items()}
        opt_params = {k: v.to(device, non_blocking=True) for k, v in opt_params.items()}
    print(f"[ewc] Loaded Fisher state from {path} → {device}")
    return fisher, opt_params


class EWCTrainer(Trainer):
    """Trainer that applies EWC as a post-backward gradient penalty."""

    def __init__(self, *args, ewc_lambda=5000.0, fisher=None, opt_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.args.ewc_lambda = ewc_lambda
        self._ewc_enabled = fisher is not None and opt_params is not None
        print(f"[ewc] EWCTrainer init: _ewc_enabled={self._ewc_enabled}, lambda={ewc_lambda}")
        self._fisher_dict = fisher or {}
        self._opt_params_dict = opt_params or {}
        self._p_list = []
        self._f_list = []
        self._o_list = []
        self._built = False
        self._chunk_size = 32

    def _build_penalty_cache(self, model):
        if self._built or not self._ewc_enabled:
            return

        param_dict = dict(model.named_parameters())
        tracked_numel = 0

        for name, fisher_tensor in self._fisher_dict.items():
            param = param_dict.get(name)
            opt_tensor = self._opt_params_dict.get(name)

            if param is None or opt_tensor is None or not param.requires_grad:
                continue

            self._p_list.append(param)
            self._f_list.append(fisher_tensor.to(param.device, dtype=param.dtype, non_blocking=True))
            self._o_list.append(opt_tensor.to(param.device, dtype=param.dtype, non_blocking=True))
            tracked_numel += param.numel()

        self._fisher_dict = {}
        self._opt_params_dict = {}
        self._built = True

        if not self._p_list:
            self._ewc_enabled = False
            print("[ewc] WARNING: no matching trainable parameters found for EWC; disabling penalty.")
            return

        state_gb = tracked_numel * self._f_list[0].element_size() * 2 / 1e9
        print(
            f"[ewc] Grad-penalty cache built: {tracked_numel:,} params across "
            f"{len(self._p_list)} tensors (~{state_gb:.2f} GB EWC state on-device)."
        )

    def _apply_ewc_grad_penalty(self):
        if not self._ewc_enabled or not self._built:
            return

        alpha = self.ewc_lambda
        if self.current_gradient_accumulation_steps > 1:
            alpha = alpha / float(self.current_gradient_accumulation_steps)

        with torch.no_grad():
            for start in range(0, len(self._p_list), self._chunk_size):
                p_chunk = []
                g_chunk = []
                f_chunk = []
                o_chunk = []

                for idx in range(start, min(start + self._chunk_size, len(self._p_list))):
                    param = self._p_list[idx]
                    if param.grad is None:
                        continue
                    p_chunk.append(param)
                    g_chunk.append(param.grad)
                    f_chunk.append(self._f_list[idx])
                    o_chunk.append(self._o_list[idx])

                if not p_chunk:
                    continue

                try:
                    diffs = torch._foreach_sub(p_chunk, o_chunk)
                    torch._foreach_mul_(diffs, f_chunk)
                    torch._foreach_add_(g_chunk, diffs, alpha=alpha)
                except RuntimeError:
                    for p, g, f, o in zip(p_chunk, g_chunk, f_chunk, o_chunk):
                        g.add_(f * (p - o), alpha=alpha)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if self._ewc_enabled:
            if not self._built:
                self._build_penalty_cache(model)
            self._apply_ewc_grad_penalty()

        return loss


def apply_proteus_attention_strategy(model, attention_mode: str):
    attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    layers = model.model.language_model.layers
    n_layers = len(layers)

    if attention_mode == "freeze":
        for layer in layers:
            for name in attn_modules:
                m = getattr(layer.self_attn, name, None)
                if m is not None:
                    m.weight.requires_grad_(False)
        print(f"[proteus] Attention: all {n_layers} layers frozen (5a).")
    elif attention_mode == "diagonal":
        freeze_up_to = n_layers // 2
        for i, layer in enumerate(layers):
            for name in attn_modules:
                m = getattr(layer.self_attn, name, None)
                if m is not None:
                    m.weight.requires_grad_(i >= freeze_up_to)
        print(f"[proteus] Attention: layers 0-{freeze_up_to-1} frozen, {freeze_up_to}-{n_layers-1} trainable (5c).")
    else:
        print(f"[proteus] Attention: all {n_layers} layers trainable (5b).")


def register_lora_fast_hooks(model, r=64, alpha=128, dropout_p=0.05):
    """LoRA on attention (q_proj, v_proj) — standard baseline."""
    import math

    for p in model.parameters():
        p.requires_grad_(False)

    lora_params = []
    lora_hooks  = []
    layers  = model.model.language_model.layers
    scaling = alpha / r
    _device = next(model.parameters()).device

    for layer in layers:
        for proj_name in ("q_proj", "v_proj"):
            if not hasattr(layer.self_attn, proj_name):
                continue

            proj = getattr(layer.self_attn, proj_name)
            inner = getattr(proj, "linear", proj)
            in_f, out_f = inner.in_features, inner.out_features

            lora_A = torch.nn.Parameter(
                torch.empty(r, in_f, dtype=torch.bfloat16, device=_device)
            )
            lora_B = torch.nn.Parameter(
                torch.zeros(out_f, r, dtype=torch.bfloat16, device=_device)
            )
            torch.nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))

            proj.register_parameter("lora_A", lora_A)
            proj.register_parameter("lora_B", lora_B)
            lora_params.extend([lora_A, lora_B])

            drop = torch.nn.Dropout(dropout_p)

            def make_hook(A, B, d):
                def hook(module, input, output):
                    x = input[0]
                    delta = torch.nn.functional.linear(d(x), B @ A) * scaling
                    return output + delta
                return hook

            lora_hooks.append(
                proj.register_forward_hook(make_hook(lora_A, lora_B, drop))
            )

    return lora_hooks, lora_params


def register_lora_ffn_hooks(model, r=64, alpha=128, dropout_p=0.05):
    """
    LoRA on FFN (gate_proj, up_proj, down_proj) — same location as Proteus.

    This ablation isolates mechanism from location. Both Proteus and LoRA-FFN
    operate on the FFN weight matrices. Proteus does full-rank updates restricted
    to the outer zone. LoRA-FFN does low-rank updates (rank r) to the full FFN
    weight matrix. The comparison answers: given the same location, does
    structured full-rank plasticity (Proteus) outperform low-rank adaptation (LoRA)?

    Rank is matched to the attention LoRA (r=64) for comparable parameter budgets.
    Note: FFN matrices are larger (gate/up: [10240, 2560], down: [2560, 10240]),
    so total LoRA-FFN parameter count is higher. This is noted in the paper.
    """
    import math

    for p in model.parameters():
        p.requires_grad_(False)

    lora_params = []
    lora_hooks  = []
    layers  = model.model.language_model.layers
    scaling = alpha / r
    _device = next(model.parameters()).device

    for layer in layers:
        mlp = layer.mlp
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, proj_name, None)
            if proj is None:
                continue

            in_f  = proj.in_features
            out_f = proj.out_features

            lora_A = torch.nn.Parameter(
                torch.empty(r, in_f, dtype=torch.bfloat16, device=_device)
            )
            lora_B = torch.nn.Parameter(
                torch.zeros(out_f, r, dtype=torch.bfloat16, device=_device)
            )
            torch.nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))

            proj.register_parameter("lora_A", lora_A)
            proj.register_parameter("lora_B", lora_B)
            lora_params.extend([lora_A, lora_B])

            drop = torch.nn.Dropout(dropout_p)

            def make_hook(A, B, d):
                def hook(module, input, output):
                    x = input[0]
                    delta = torch.nn.functional.linear(d(x), B @ A) * scaling
                    return output + delta
                return hook

            lora_hooks.append(
                proj.register_forward_hook(make_hook(lora_A, lora_B, drop))
            )

    trainable = sum(p.numel() for p in lora_params)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[lora_ffn] Registered {len(lora_hooks)} forward hooks on FFN (gate/up/down).")
    print(f"[lora_ffn] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    return lora_hooks, lora_params


def setup_proteus_condition(model, args):
    model.train()
    hooks = register_hooks(model)
    apply_proteus_attention_strategy(model, args.attention)
    return hooks, Trainer, None, None


def setup_full_condition(model, args):
    model.train()
    print("[full] All parameters trainable.")
    return [], Trainer, None, None


def setup_lora_condition(model, args):
    lora_hooks, lora_params = register_lora_fast_hooks(model)
    trainable = sum(p.numel() for p in lora_params)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[lora] Registered {len(lora_hooks)} forward hooks (attention: q/v).")
    print(f"[lora] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    return [], Trainer, None, None


def setup_lora_ffn_condition(model, args):
    lora_hooks, lora_params = register_lora_ffn_hooks(model)
    return [], Trainer, None, None


def setup_ewc_condition(model, args):
    model.train()
    fisher = None
    opt_params = None
    if args.ewc_state:
        device = next(model.parameters()).device
        fisher, opt_params = load_ewc_state(args.ewc_state, device=device)
        print(f"[ewc] lambda={args.ewc_lambda}, prior state loaded to {device}.")
    else:
        print("[ewc] No prior state — first domain, training without penalty.")
    return [], EWCTrainer, fisher, opt_params


def setup_replay_condition(model, args):
    model.train()
    if not args.replay_buffer:
        print("[replay] WARNING: no --replay_buffer provided. Running as full fine-tune.")
    print(f"[replay] ratio={args.replay_ratio}")
    return [], Trainer, None, None


def setup_training_condition(model, args):
    handlers = {
        "proteus":  setup_proteus_condition,
        "full":     setup_full_condition,
        "lora":     setup_lora_condition,
        "lora_ffn": setup_lora_ffn_condition,
        "ewc":      setup_ewc_condition,
        "replay":   setup_replay_condition,
    }
    return handlers[args.condition](model, args)


def maybe_compile_model(model, enable_compile: bool):
    if not enable_compile:
        return model

    print("Compiling model with Triton (CUDA graphs disabled)...")
    try:
        from torch import _dynamo
        from torch._inductor import config as inductor_config

        _dynamo.config.suppress_errors = True
        if hasattr(inductor_config.triton, "cudagraphs"):
            inductor_config.triton.cudagraphs = False
        if hasattr(inductor_config.triton, "cudagraph_trees"):
            inductor_config.triton.cudagraph_trees = False
        return torch.compile(
            model,
            mode="max-autotune",
            backend="inductor",
        )
    except Exception as exc:
        print(f"[compile] WARNING: torch.compile failed ({type(exc).__name__}: {exc})")
        print("[compile] WARNING: Falling back to eager mode.")
        return model


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
def load_domain(domain: str, split: str = "train") -> Dataset:
    path = DATA_DIR / domain / f"{split}.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def load_replay_buffer(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[replay] Loaded {len(records)} records from {path}")
    return records


RESPONSE_KEY = "### Response:\n"


def format_prompt_parts(row: dict) -> tuple[str, str]:
    instruction = row["instruction"]
    inp         = row.get("input", "").strip()
    output      = row["output"]
    if inp:
        prefix = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"{RESPONSE_KEY}"
        )
    else:
        prefix = (
            f"### Instruction:\n{instruction}\n\n"
            f"{RESPONSE_KEY}"
        )
    return prefix, output


def tokenize_dataset(raw_ds: Dataset, tokenizer) -> Dataset:
    def tokenize_fn(batch):
        all_input_ids = []
        all_labels    = []
        all_lengths   = []

        for i in range(len(batch["instruction"])):
            prefix, response = format_prompt_parts({
                "instruction": batch["instruction"][i],
                "input":       batch.get("input", [""] * len(batch["instruction"]))[i],
                "output":      batch["output"][i],
            })

            prefix_ids = tokenizer(
                prefix,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            full_ids = tokenizer(
                prefix + response,
                truncation=True,
                max_length=MAX_LENGTH,
                add_special_tokens=True,
            )["input_ids"]

            n_prefix = min(len(prefix_ids), len(full_ids) - 1)
            labels   = [-100] * n_prefix + full_ids[n_prefix:]

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_lengths.append(len(full_ids))

        return {"input_ids": all_input_ids, "labels": all_labels, "length": all_lengths}

    return raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",       required=True,
                        choices=["medical", "legal", "code", "multilingual"])
    parser.add_argument("--condition",    required=True,
                        choices=["proteus", "full", "lora", "lora_ffn", "ewc", "replay"])
    parser.add_argument("--start_from",   type=str, default=None)
    parser.add_argument("--max_steps",    type=int,   default=500)
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--grad_accum",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--ewc_lambda",   type=float, default=5000.0)
    parser.add_argument("--ewc_state",    type=str,   default=None)
    parser.add_argument("--ewc_samples",  type=int,   default=200)
    parser.add_argument("--replay_buffer",type=str,   default=None)
    parser.add_argument("--replay_ratio", type=float, default=0.3)
    parser.add_argument("--attention",    type=str,   default="train",
                        choices=["train", "freeze", "diagonal"])
    parser.add_argument("--compile",      action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--out_dir",      type=str,   default=None)
    parser.add_argument("--status_file",  type=str,   default=None)
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = Path(args.out_dir) if args.out_dir else CKPT_DIR / args.condition / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    status_writer = StatusWriter(args.status_file)
    status_writer.emit({
        "phase": "train",
        "state": "initializing",
        "condition": args.condition,
        "domain": args.domain,
        "step": 0,
        "total_steps": args.max_steps,
        "it_s": 0.0,
        "eta_s": None,
    })

    print(f"\n=== Proteus training ===")
    print(f"  Domain:    {args.domain}")
    print(f"  Condition: {args.condition}")
    if args.condition == "proteus":
        print(f"  Attention: {args.attention}")
    print(f"  Compile:   {args.compile}")
    print(f"  Output:    {out_dir}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _model_source = args.start_from if args.start_from else MODEL_ID
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        _model_source,
        torch_dtype=torch.bfloat16,
        device_map={"" : _local_rank},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model.config.use_cache = False

    print("Amputating vision and audio encoders...")
    if hasattr(model, "vision_tower"):
        del model.vision_tower
    if hasattr(model, "audio_encoder"):
        del model.audio_encoder
    if hasattr(model, "multi_modal_projector"):
        del model.multi_modal_projector
    torch.cuda.empty_cache()

    apply_aiter_optimizations(model)

    hooks, trainer_class, fisher, opt_params = setup_training_condition(model, args)

    model = maybe_compile_model(model, args.compile)

    print(f"Loading {args.domain} dataset...")
    raw_ds = load_domain(args.domain, split="train")

    effective_batch = args.batch_size * args.grad_accum
    max_samples = args.max_steps * effective_batch * 2

    if args.condition == "replay" and args.replay_buffer:
        replay_records = load_replay_buffer(args.replay_buffer)
        # Truncate the domain dataset FIRST so replay data isn't amputated
        if len(raw_ds) > max_samples:
            print(f"[data] Truncating domain to {max_samples:,} before replay injection.")
            raw_ds = raw_ds.select(range(max_samples))
        n_replay = int(len(raw_ds) * args.replay_ratio / (1 - args.replay_ratio))
        n_replay = min(n_replay, len(replay_records))
        sampled  = random.sample(replay_records, n_replay)
        combined_list = list(raw_ds) + sampled
        random.shuffle(combined_list)
        raw_ds = Dataset.from_list(combined_list)
        print(f"[replay] {len(raw_ds) - n_replay} current + {n_replay} replay = {len(raw_ds)} total")
    else:
        if len(raw_ds) > max_samples:
            print(f"[data] Truncating {len(raw_ds):,} → {max_samples:,} samples (2x needed)")
            raw_ds = raw_ds.select(range(max_samples))

    tokenized = tokenize_dataset(raw_ds, tokenizer)
    collator  = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=128)

    training_args = TrainingArguments(
        output_dir               = str(out_dir),
        max_steps                = args.max_steps,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        gradient_checkpointing   = args.gradient_checkpointing,
        learning_rate            = args.lr,
        lr_scheduler_type        = "cosine",
        warmup_steps             = 25,
        bf16                     = True,
        optim                    = "adamw_torch_fused",
        logging_steps            = 10,
        save_strategy            = "no",
        report_to                = "none",
        dataloader_num_workers   = 4,
        dataloader_pin_memory    = True,
        train_sampling_strategy  = "group_by_length",
        length_column_name       = "length",
        remove_unused_columns    = True,
    )

    trainer_kwargs = dict(
        model         = model,
        args          = training_args,
        train_dataset = tokenized,
        data_collator = collator,
    )
    if args.condition == "ewc":
        trainer_kwargs.update(
            ewc_lambda  = args.ewc_lambda,
            fisher      = fisher,
            opt_params  = opt_params,
        )

    _rank = int(os.environ.get("RANK", "0"))

    trainer = trainer_class(**trainer_kwargs)
    trainer.add_callback(
        TrainStatusCallback(
            writer=status_writer,
            condition=args.condition,
            domain=args.domain,
            total_steps=args.max_steps,
        )
    )

    print("--- VRAM AUDIT ---")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print("------------------")

    try:
        print("Training...")
        trainer.train()

        status_writer.emit({
            "phase": "train",
            "state": "saving",
            "condition": args.condition,
            "domain": args.domain,
            "step": args.max_steps,
            "total_steps": args.max_steps,
            "it_s": 0.0,
            "eta_s": 0.0,
        })

        for h in hooks:
            h.remove()

        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        print("[cleanup] Destroying HF Trainer and freeing optimizer VRAM...")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        if args.condition == "ewc" and _rank == 0:
            print("[ewc] Computing Fisher matrix for next domain...")
            new_fisher, new_opt = compute_fisher(model, tokenized, n_samples=args.ewc_samples)
            save_ewc_state(new_fisher, new_opt, out_dir)

        status_writer.emit({
            "phase": "train",
            "state": "done",
            "condition": args.condition,
            "domain": args.domain,
            "step": args.max_steps,
            "total_steps": args.max_steps,
            "it_s": 0.0,
            "eta_s": 0.0,
        })
        print(f"\nDone. Checkpoint saved to {out_dir}")
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        try:
            del trainer
            del model
            del tokenized
            del collator
        except Exception:
            pass
        cleanup_runtime()


if __name__ == "__main__":
    main()
