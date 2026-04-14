"""
Proteus — Fine-tune Wrapper
Trains Gemma 4 E4B on one domain under a specified condition.

Conditions:
  proteus   — Core frozen via gradient hook (the thesis)
  full      — Full fine-tune, no protection
  lora      — LoRA adapters only, base model frozen entirely

Usage:
  python train.py --domain medical --condition proteus
  python train.py --domain legal   --condition full   --max_steps 200
  python train.py --domain code    --condition lora
  python train.py --domain code    --condition proteus --compile   # Triton kernels
"""

import argparse
import json
from pathlib import Path

import torch
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
# Dataset
# ─────────────────────────────────────────────
def load_domain(domain: str) -> Dataset:
    path = DATA_DIR / domain / "train.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


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


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",     required=True,
                        choices=["medical", "legal", "code", "multilingual"])
    parser.add_argument("--condition",  required=True,
                        choices=["proteus", "full", "lora"])
    parser.add_argument("--max_steps",  type=int,   default=500)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--grad_accum", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--compile",    action="store_true",
                        help="Compile model with torch.compile (Triton). "
                             "Adds ~1-2 min warm-up on first run, faster thereafter.")
    args = parser.parse_args()

    out_dir = CKPT_DIR / args.condition / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Proteus training ===")
    print(f"  Domain:    {args.domain}")
    print(f"  Condition: {args.condition}")
    print(f"  Compile:   {args.compile}")
    print(f"  Output:    {out_dir}\n")

    # ── Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model
    # sdpa: PyTorch built-in fused attention, no head dim restriction, Blackwell-compatible.
    # flash_attention_2 is excluded: Gemma 4 has layers with head_dim > 256, FA2 hard limit.
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # ── Condition setup
    hooks = []
    if args.condition == "proteus":
        model.train()
        hooks = register_hooks(model)

    elif args.condition == "full":
        model.train()
        print("[full] All parameters trainable.")

    elif args.condition == "lora":
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ── Triton compile (opt-in)
    # Gradient hooks survive compilation: they fire on the gradient tensor
    # after the backward pass, outside the compiled graph.
    # Skip for smoke tests (--max_steps < 50) since warm-up cost outweighs benefit.
    if args.compile:
        if args.max_steps < 50:
            print("[compile] Skipped — max_steps < 50, warm-up cost not worth it.")
        else:
            print("[compile] Compiling model with torch.compile (mode=reduce-overhead)...")
            model = torch.compile(model, mode="reduce-overhead")
            print("[compile] Done. First batch will trigger Triton kernel compilation.")

    # ── Dataset
    print(f"Loading {args.domain} dataset...")
    raw_ds = load_domain(args.domain)

    def tokenize_fn(batch):
        texts = [format_prompt({
            "instruction": batch["instruction"][i],
            "input":       batch.get("input", [""] * len(batch["instruction"]))[i],
            "output":      batch["output"][i],
        }) for i in range(len(batch["instruction"]))]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_ds.column_names,
    )

    # ── Training
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,        # recompute activations, saves VRAM
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        optim="adamw_torch_fused",          # fused Adam: no bitsandbytes, works on Blackwell
        logging_steps=10,
        save_steps=args.max_steps,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=2,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("Training...")
    trainer.train()

    # ── Cleanup hooks before save
    for h in hooks:
        h.remove()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nDone. Checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()
