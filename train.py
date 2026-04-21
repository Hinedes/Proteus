"""
Proteus — Fine-tune Wrapper
Trains Gemma 4 E4B on one domain under a specified condition.

Conditions:
  proteus   — Core frozen via gradient hook (the thesis)
  full      — Full fine-tune, no protection
  lora      — LoRA adapters only, base model frozen entirely
  ewc       — Elastic Weight Consolidation (importance-weighted regularization)
  replay    — Data replay from previous domains

Usage:
  python train.py --domain medical --condition proteus
  python train.py --domain legal   --condition full   --max_steps 200
  python train.py --domain code    --condition lora
  python train.py --domain code    --condition proteus --compile

  # EWC: first domain needs no prior state; subsequent domains load it
  python train.py --domain medical --condition ewc
  python train.py --domain legal   --condition ewc --ewc_state checkpoints/ewc/medical/fisher.pt

  # Replay: pass a .jsonl file of buffered prior-domain examples
  python train.py --domain legal   --condition replay --replay_buffer data/replay_buffer.jsonl
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
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

# ─────────────────────────────────────────────
# Gradient hook (Proteus condition)
# ─────────────────────────────────────────────
def register_hooks(model):
    hooks = []
    layers = model.model.language_model.layers

    def make_hook(proj_type):
        def hook(grad):
            g = grad.clone()
            if proj_type in ("gate", "up"):
                g[:CORE_MID, :CORE_HIDDEN] = 0.0
            else:
                g[:CORE_HIDDEN, :CORE_MID] = 0.0
            return g
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
def compute_fisher(model, dataset, tokenizer, n_samples=200, batch_size=4):
    """
    Diagonal Fisher estimate via squared gradients on a sample of the dataset.
    Batched: processes batch_size samples per backward pass instead of 1.
    Returns {param_name: fisher_diagonal_tensor}.
    """
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

    for n in fisher:
        fisher[n] /= n_batches

    model.train()
    return fisher, opt_params


def save_ewc_state(fisher, opt_params, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"fisher": fisher, "opt_params": opt_params}, path / "fisher.pt")
    print(f"[ewc] Fisher state saved to {path / 'fisher.pt'}")


def load_ewc_state(path: str):
    state = torch.load(path, map_location="cpu")
    print(f"[ewc] Loaded Fisher state from {path}")
    return state["fisher"], state["opt_params"]


class EWCTrainer(Trainer):
    """Trainer that adds EWC penalty to the standard cross-entropy loss.

    Optimization: pre-flattens fisher and opt_params into single tensors on
    first step so the penalty is one vectorized op instead of a Python loop
    over thousands of named parameters every step.
    """

    def __init__(self, *args, ewc_lambda=5000.0, fisher=None, opt_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda       = ewc_lambda
        self._fisher_dict     = fisher      # {name: tensor}, freed after first step
        self._opt_params_dict = opt_params  # {name: tensor}, freed after first step
        # Pre-flattened tensors built on first compute_loss call
        self._param_names  = None   # ordered list of param names in fisher
        self._fisher_flat  = None   # [N] tensor
        self._opt_flat     = None   # [N] tensor

    def _build_flat(self, model):
        """Align fisher/opt to current model params and flatten. Called once."""
        names, f_parts, o_parts = [], [], []
        dev = next(model.parameters()).device
        for n, p in model.named_parameters():
            if n in self._fisher_dict:
                names.append(n)
                f_parts.append(self._fisher_dict[n].to(dev).reshape(-1))
                o_parts.append(self._opt_params_dict[n].to(dev).reshape(-1))
        self._param_names = names
        self._fisher_flat = torch.cat(f_parts) if f_parts else None
        self._opt_flat    = torch.cat(o_parts) if o_parts else None
        # Free dicts -- no longer needed
        self._fisher_dict = None
        self._opt_params_dict = None
        print(f"[ewc] Penalty tensor built: {self._fisher_flat.numel():,} params tracked.")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss    = outputs.loss

        if self._fisher_dict is not None:
            # First step: build flattened tensors
            self._build_flat(model)

        if self._fisher_flat is not None:
            # Single vectorized op: no Python loop
            p_flat = torch.cat([
                dict(model.named_parameters())[n].reshape(-1)
                for n in self._param_names
            ])
            ewc_penalty = (self._fisher_flat * (p_flat - self._opt_flat) ** 2).sum()
            loss = loss + (self.ewc_lambda / 2.0) * ewc_penalty

        return (loss, outputs) if return_outputs else loss


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


def format_prompt(row: dict) -> str:
    instruction = row["instruction"]
    inp         = row.get("input", "").strip()
    output      = row["output"]
    if inp:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


RESPONSE_KEY = "### Response:\n"


def format_prompt_parts(row: dict) -> tuple[str, str]:
    """Returns (prompt_prefix, response_text) separately for label masking."""
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

        for i in range(len(batch["instruction"])):
            prefix, response = format_prompt_parts({
                "instruction": batch["instruction"][i],
                "input":       batch.get("input", [""] * len(batch["instruction"]))[i],
                "output":      batch["output"][i],
            })

            # Tokenize prefix and full sequence separately to find boundary
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

            # Mask prompt tokens with -100 so loss is only on response
            # Guard: ensure at least 1 valid label token survives truncation.
            # Without this, long-prefix datasets (e.g. legal) produce sequences
            # where prefix alone exceeds MAX_LENGTH, masking all labels to -100
            # and causing NaN loss that poisons the optimizer state.
            n_prefix = min(len(prefix_ids), len(full_ids) - 1)
            labels   = [-100] * n_prefix + full_ids[n_prefix:]

            all_input_ids.append(full_ids)
            all_labels.append(labels)

        return {"input_ids": all_input_ids, "labels": all_labels}

    return raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",       required=True,
                        choices=["medical", "legal", "code", "multilingual"])
    parser.add_argument("--condition",    required=True,
                        choices=["proteus", "full", "lora", "ewc", "replay"])
    parser.add_argument("--max_steps",    type=int,   default=500)
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--grad_accum",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--ewc_lambda",   type=float, default=5000.0,
                        help="EWC regularization strength.")
    parser.add_argument("--ewc_state",    type=str,   default=None,
                        help="Path to fisher.pt from previous domain (EWC only).")
    parser.add_argument("--ewc_samples",  type=int,   default=200,
                        help="Samples used to estimate Fisher diagonal.")
    parser.add_argument("--replay_buffer",type=str,   default=None,
                        help="Path to .jsonl replay buffer (Replay only).")
    parser.add_argument("--replay_ratio", type=float, default=0.3,
                        help="Fraction of each batch drawn from replay buffer.")
    parser.add_argument("--attention",    type=str,   default="train",
                        choices=["train", "freeze", "diagonal"],
                        help=(
                            "Attention strategy for Proteus condition. "
                            "'train': all attention layers update (default, Condition 5b). "
                            "'freeze': all attention frozen (Condition 5a). "
                            "'diagonal': freeze bottom half, train top half (Condition 5c)."
                        ))
    parser.add_argument("--compile",      action="store_true",
                        help="Compile with torch.compile (Triton). ~1-2 min warm-up.")
    args = parser.parse_args()

    out_dir = CKPT_DIR / args.condition / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Proteus training ===")
    print(f"  Domain:    {args.domain}")
    print(f"  Condition: {args.condition}")
    if args.condition == "proteus":
        print(f"  Attention: {args.attention}")
    print(f"  Compile:   {args.compile}")
    print(f"  Output:    {out_dir}\n")

    # ── Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model
    # sdpa: PyTorch built-in fused attention, no head dim restriction, Blackwell-compatible.
    # flash_attention_2 excluded: Gemma 4 has head_dim > 256, FA2 hard limit.
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # ── Condition setup
    hooks   = []
    fisher  = None
    opt_params = None
    trainer_class = Trainer

    if args.condition == "proteus":
        model.train()
        hooks = register_hooks(model)
        # ── Attention strategy (Proteus only)
        attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        layers = model.model.language_model.layers
        n_layers = len(layers)
        if args.attention == "freeze":
            # Condition 5a: freeze all attention across all layers
            for layer in layers:
                for name in attn_modules:
                    m = getattr(layer.self_attn, name, None)
                    if m is not None:
                        m.weight.requires_grad_(False)
            print(f"[proteus] Attention: all {n_layers} layers frozen (5a).")
        elif args.attention == "diagonal":
            # Condition 5c: freeze bottom half, train top half
            freeze_up_to = n_layers // 2
            for i, layer in enumerate(layers):
                for name in attn_modules:
                    m = getattr(layer.self_attn, name, None)
                    if m is not None:
                        m.weight.requires_grad_(i >= freeze_up_to)
            print(f"[proteus] Attention: layers 0-{freeze_up_to-1} frozen, {freeze_up_to}-{n_layers-1} trainable (5c).")
        else:
            print(f"[proteus] Attention: all {n_layers} layers trainable (5b).")

    elif args.condition == "full":
        model.train()
        print("[full] All parameters trainable.")

    elif args.condition == "lora":
        # Fast LoRA via forward hooks -- same pattern as Proteus gradient hooks.
        # Avoids module-wrapping overhead. Caches merged BA so each hook is
        # one fused matmul instead of two sequential ones.
        import math

        for p in model.parameters():
            p.requires_grad_(False)

        lora_params = []
        lora_hooks  = []
        layers  = model.model.language_model.layers
        r, alpha, dropout_p = 16, 32, 0.05
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

        trainable = sum(p.numel() for p in lora_params)
        total     = sum(p.numel() for p in model.parameters())
        print(f"[lora] Registered {len(lora_hooks)} forward hooks (fast path).")
        print(f"[lora] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    elif args.condition == "ewc":
        model.train()
        trainer_class = EWCTrainer
        if args.ewc_state:
            fisher, opt_params = load_ewc_state(args.ewc_state)
            print(f"[ewc] lambda={args.ewc_lambda}, prior state loaded.")
        else:
            print("[ewc] No prior state — first domain, training without penalty.")

    elif args.condition == "replay":
        model.train()
        if not args.replay_buffer:
            print("[replay] WARNING: no --replay_buffer provided. Running as full fine-tune.")
        print(f"[replay] ratio={args.replay_ratio}")

    # ── Triton compile (opt-in)
    if args.compile:
        if args.max_steps < 50:
            print("[compile] Skipped — max_steps < 50, warm-up cost not worth it.")
        else:
            print("[compile] Compiling model with torch.compile (mode=reduce-overhead)...")
            model = torch.compile(model, mode="reduce-overhead")
            print("[compile] Done. First batch will trigger Triton kernel compilation.")

    # ── Dataset
    print(f"Loading {args.domain} dataset...")
    raw_ds = load_domain(args.domain, split="train")

    if args.condition == "replay" and args.replay_buffer:
        replay_records = load_replay_buffer(args.replay_buffer)
        # Interleave replay samples proportionally
        n_replay = int(len(raw_ds) * args.replay_ratio / (1 - args.replay_ratio))
        n_replay = min(n_replay, len(replay_records))
        sampled  = random.sample(replay_records, n_replay)
        combined = Dataset.from_list(list(raw_ds) + sampled)
        print(f"[replay] {len(raw_ds)} current + {n_replay} replay = {len(combined)} total")
        raw_ds = combined

    tokenized = tokenize_dataset(raw_ds, tokenizer)
    collator  = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    # ── Training args
    training_args = TrainingArguments(
        output_dir               = str(out_dir),
        max_steps                = args.max_steps,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate            = args.lr,
        lr_scheduler_type        = "cosine",
        warmup_steps             = 25,
        bf16                     = True,
        optim                    = "adamw_torch_fused",
        logging_steps            = 10,
        save_steps               = args.max_steps,   # save once at end
        save_total_limit         = 1,
        report_to                = "none",
        dataloader_num_workers   = 4,
        dataloader_pin_memory    = True,
        remove_unused_columns    = False,
    )

    # ── Trainer
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

    trainer = trainer_class(**trainer_kwargs)

    print("Training...")
    trainer.train()

    # ── EWC: compute and save Fisher state for next domain
    if args.condition == "ewc":
        print("[ewc] Computing Fisher matrix for next domain...")
        new_fisher, new_opt = compute_fisher(model, tokenized, tokenizer, n_samples=args.ewc_samples)
        save_ewc_state(new_fisher, new_opt, out_dir)

    # ── Cleanup hooks before save
    for h in hooks:
        h.remove()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nDone. Checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()