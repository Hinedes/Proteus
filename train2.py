#!/usr/bin/env python3
"""
train_msf_v2.py — Matryoshka Subspace Freezing V2, safe for MI300X.
All proven features from MSF1 (train.py) now merged in.

Usage:
  # MSF‑2 normal run
  python train2.py --domain medical --out_dir /checkpoints/medical --seed 1

  # Full fine‑tune baseline
  python train2.py --domain medical --full_ft --out_dir /checkpoints/full_medical

  # With gradient checkpointing + optional compile
  python train2.py --domain medical --gradient_checkpointing --compile --out_dir ...

  # Experiment with aiter fused kernel (if available)
  python train2.py --domain medical --aiter --out_dir ...
"""

import argparse, atexit, gc, json, logging, os, random, sys, time, warnings
from pathlib import Path
from functools import partial

import torch
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import BatchSampler, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import seed_worker

# ── Silence third‑party noise ──
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformer_engine").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# ── Constants ──
MODEL_ID       = "google/gemma-4-E4B-it"
CORE_HIDDEN    = 2560   # full hidden dimension
CORE_MID       = 10240  # full intermediate dimension
MAX_LENGTH     = 512

# ── Corrected coupling scores (clean per‑layer isolation) ──
DEFAULT_COUPLING = {
    0: 0.8906, 1: 0.8984, 2: 0.9180, 3: 0.8711, 4: 0.8984,
    5: 0.9180, 6: 0.9102, 7: 0.8828, 8: 0.8945, 9: 0.8906,
    10: 0.8789, 11: 0.8906, 12: 0.9219, 13: 0.9141, 14: 0.8477,
    15: 0.8711, 16: 0.8711, 17: 0.8555, 18: 0.8750, 19: 0.8789,
    20: 0.8828, 21: 0.8633, 22: 0.8945, 23: 0.8828, 24: 0.8359,
    25: 0.8086, 26: 0.7969, 27: 0.8008, 28: 0.8359, 29: 0.8633,
    30: 0.8867, 31: 0.8516, 32: 0.8828, 33: 0.8672, 34: 0.8672,
    35: 0.8398, 36: 0.8281, 37: 0.8125, 38: 0.8203, 39: 0.8398,
    40: 0.8398, 41: 0.8477,
}

# ── Cleanup (from train.py) ──
def cleanup_runtime():
    try:
        children = torch.multiprocessing.active_children()
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

# ── aiter (optional, from train.py) ──
def apply_aiter_optimizations(model):
    try:
        import aiter
    except ImportError:
        print("[aiter] aiter not installed – skipping.")
        return

    try:
        layers = model.model.language_model.layers
        patched = 0

        def make_aiter_mlp_forward(original_mlp):
            def forward(hidden_states):
                gate = original_mlp.gate_proj(hidden_states)
                up   = original_mlp.up_proj(hidden_states)
                fused_input = torch.cat([gate, up], dim=-1)
                activated   = torch.empty_like(gate)
                aiter.silu_and_mul(activated, fused_input)
                return original_mlp.down_proj(activated)
            return forward

        for layer in layers:
            mlp = layer.mlp
            mlp.forward = make_aiter_mlp_forward(mlp)
            patched += 1

        print(f"[aiter] Patched {patched} MLP layers with fused silu_and_mul.")
    except Exception as e:
        print(f"[aiter] Skipping optimizations: {e}")

# ── Coupling helpers ──
def load_coupling(csv_path=None):
    if csv_path:
        import csv
        scores = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                layer = int(row.get("layer", row.get("Layer")))
                score = float(row.get("coupling", row.get("s_l")))
                scores[layer] = score
        return scores
    return DEFAULT_COUPLING

def per_layer_freeze_dims(coupling):
    dims = {}
    for layer, s in coupling.items():
        f = s ** 2
        dims[layer] = (round(f * CORE_MID), round(f * CORE_HIDDEN))
    return dims

class GradientNormRecorder:
    def __init__(self, layer_idx, proj_type, m, n):
        self.layer_idx = layer_idx
        self.proj_type = proj_type
        self.m, self.n = m, n
        self.core_norm = None
        self.shell_norm = None
        self.done = False

    def hook(self, grad):
        if self.done: return grad
        with torch.no_grad():
            g = grad.detach()
            if self.proj_type != "down":
                core = g[:self.m, :self.n]
                shell = torch.cat([g[self.m:, :].flatten(), g[:self.m, self.n:].flatten()])
            else:
                core = g[:self.n, :self.m]
                shell = torch.cat([g[self.n:, :].flatten(), g[:self.n, self.m:].flatten()])
            self.core_norm = core.norm().item()
            self.shell_norm = shell.norm().item()
        self.done = True
        return grad

def register_v2_hooks(model, coupling, record_grad_norms=False):
    layers = model.model.language_model.layers
    dims = per_layer_freeze_dims(coupling)
    all_hooks = []
    norm_recorders = {}

    for i, layer in enumerate(layers):
        m, n = dims.get(i, (CORE_MID, CORE_HIDDEN))

        def make_mask_hook(proj, _m, _n):
            def hook(grad):
                if proj in ("gate", "up"):
                    grad[:_m, :_n] = 0.0
                else:
                    grad[:_n, :_m] = 0.0
                return grad
            return hook

        all_hooks.append(layer.mlp.gate_proj.weight.register_hook(make_mask_hook("gate", m, n)))
        all_hooks.append(layer.mlp.up_proj.weight.register_hook(make_mask_hook("up", m, n)))
        all_hooks.append(layer.mlp.down_proj.weight.register_hook(make_mask_hook("down", m, n)))

        if record_grad_norms:
            for proj in ("gate", "up", "down"):
                param = getattr(layer.mlp, f"{proj}_proj").weight
                rec = GradientNormRecorder(i, proj, m, n)
                param.register_hook(rec.hook)
                norm_recorders[(i, proj)] = rec

    print(f"[msf_v2] {len(all_hooks)} gradient mask hooks registered.")
    return all_hooks, norm_recorders

# ── Data loading ──
DATA_DIR = Path("data")
RESPONSE_KEY = "### Response:\n"

def load_domain(domain, split="train"):
    path = DATA_DIR / domain / f"{split}.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)

def format_prompt(row):
    instr = row["instruction"]
    inp = row.get("input", "").strip()
    out = row["output"]
    prefix = f"### Instruction:\n{instr}\n\n"
    if inp:
        prefix += f"### Input:\n{inp}\n\n"
    prefix += RESPONSE_KEY
    return prefix, out

def tokenize_dataset(raw_ds, tokenizer):
    def tokenize_fn(batch):
        input_ids_list, labels_list, lengths_list = [], [], []
        for i in range(len(batch["instruction"])):
            prefix, response = format_prompt({
                "instruction": batch["instruction"][i],
                "input": batch.get("input", [""]*len(batch["instruction"]))[i],
                "output": batch["output"][i],
            })
            prefix_ids = tokenizer(prefix, truncation=False, add_special_tokens=True)["input_ids"]
            full_ids = tokenizer(prefix + response, truncation=True, max_length=MAX_LENGTH,
                                 add_special_tokens=True)["input_ids"]
            n_prefix = min(len(prefix_ids), len(full_ids) - 1)
            labels = full_ids.copy()
            labels[:n_prefix] = [-100] * n_prefix
            input_ids_list.append(full_ids)
            labels_list.append(labels)
            lengths_list.append(len(full_ids))
        return {"input_ids": input_ids_list, "labels": labels_list, "length": lengths_list}
    return raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names)

# ── Status writers (from train.py) ──
class StatusWriter:
    def __init__(self, path):
        self.path = Path(path) if path else None
    def emit(self, payload):
        if self.path is None: return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = dict(payload); data["ts"] = time.time()
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f: json.dump(data, f)
            os.replace(tmp, self.path)
        except: pass

class ProgressWriter:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = []
    def record(self, entry):
        self.data.append(entry)
        with open(self.path, "w") as f: json.dump(self.data, f, indent=2)

# ── Native length‑grouped sampler ──
class NativeLengthGroupedBatchSampler(BatchSampler):
    def __init__(self, lengths, batch_size, seed=42, drop_last=False):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda index: self.lengths[index])
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(batches)
        self._epoch += 1
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

class NativeLengthGroupedTrainer(Trainer):
    def __init__(self, *args, train_batch_sampler=None, **kwargs):
        self._train_batch_sampler = train_batch_sampler
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self._train_batch_sampler is None:
            return super().get_train_dataloader()
        dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description="training")
        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1
        dataloader_kwargs = {
            "batch_sampler": self._train_batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
        }
        if self.args.dataloader_num_workers > 0:
            if self.args.dataloader_prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
            dataloader_kwargs["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_kwargs))

# ── Callback with pinned tqdm bar ──
class TrainStatusCallback(TrainerCallback):
    def __init__(self, writer, domain, total_steps):
        self.writer = writer
        self.domain = domain
        self.total_steps = total_steps
        self.start_time = None
        self.last_loss = None
        self.pbar = None

    def _compute_stats(self, step):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = max(time.time() - self.start_time, 1e-6)
        it_s = step / elapsed if step > 0 else 0.0
        eta_s = None
        if self.total_steps > 0 and step < self.total_steps and it_s > 0:
            eta_s = (self.total_steps - step) / it_s
        return it_s, eta_s

    def _emit(self, state, step):
        it_s, eta_s = self._compute_stats(step)
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
        self.pbar = tqdm(
            total=self.total_steps,
            position=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        self._emit("running", 0)

    def on_step_end(self, args, state, control, **kwargs):
        self._emit("running", state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if isinstance(logs, dict) and logs.get("loss") is not None:
            self.last_loss = logs["loss"]
        step = state.global_step
        it_s, eta_s = self._compute_stats(step)
        if self.pbar is not None:
            self.pbar.n = step
            postfix = {
                "loss": f"{self.last_loss:.4f}" if self.last_loss is not None else "N/A",
                "it/s": f"{it_s:.2f}",
                "stage": "train",
                "domain": self.domain,
            }
            if eta_s is not None:
                postfix["ETA"] = f"{eta_s:.0f}s" if eta_s < 120 else f"{eta_s / 60:.1f}min"
            self.pbar.set_postfix(**postfix)
            self.pbar.refresh()
        self._emit("running", state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        self._emit("done", state.global_step)

# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["medical","legal","code"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_from", default=None)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--coupling_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--status_file", default=None)
    parser.add_argument("--progress_file", default="progress.json")
    parser.add_argument("--record_grad_norms", action="store_true")
    parser.add_argument("--full_ft", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--aiter", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    status_writer = StatusWriter(args.status_file)
    status_writer.emit({"phase":"train","state":"initializing","condition":"msf_v2",
                        "domain":args.domain,"step":0,"total_steps":args.max_steps})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_path = args.start_from if args.start_from else MODEL_ID
    print(f"Loading model from {load_path}")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # Remove useless towers (exactly as in train.py)
    for attr in ["vision_tower", "audio_encoder", "multi_modal_projector"]:
        if hasattr(model, attr):
            delattr(model, attr)
    torch.cuda.empty_cache()

    # Optional aiter fused kernel
    if args.aiter:
        apply_aiter_optimizations(model)
    else:
        print("[msf_v2] Using standard PyTorch MLP (aiter disabled).")

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[msf_v2] Gradient checkpointing enabled.")

    # Register hooks or full FT
    if args.full_ft:
        hooks = []
        norm_recorders = {}
        print("[full_ft] All parameters trainable, no mask applied.")
    else:
        coupling = load_coupling(args.coupling_csv)
        hooks, norm_recorders = register_v2_hooks(model, coupling, args.record_grad_norms)
        print("[msf_v2] Attention layers fully trainable.")

    # Compile (optional, dynamic=True because batch sizes vary)
    if args.compile:
        try:
            print("[msf_v2] torch.compile enabled (dynamic=True).")
            model = torch.compile(model, dynamic=True, mode="reduce-overhead")
        except Exception as e:
            print(f"[msf_v2] torch.compile failed ({e}); continuing eagerly.")

    raw_ds = load_domain(args.domain, "train")
    max_samples = args.max_steps * args.batch_size * 2
    if len(raw_ds) > max_samples:
        raw_ds = raw_ds.select(range(max_samples))
    tokenized = tokenize_dataset(raw_ds, tokenizer)
    lengths = tokenized["length"]
    tokenized = tokenized.remove_columns("length")
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True,
                                      pad_to_multiple_of=128)

    sampler = NativeLengthGroupedBatchSampler(lengths=lengths, batch_size=args.batch_size, seed=args.seed)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=25,
        bf16=True,
        optim="adamw_torch",
        disable_tqdm=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=8,            # 8 is safe, you can bump to 16
        dataloader_pin_memory=True,
        remove_unused_columns=False,         # we handle length column ourselves
    )

    trainer = NativeLengthGroupedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        train_batch_sampler=sampler,
    )
    trainer.add_callback(TrainStatusCallback(status_writer, args.domain, args.max_steps))

    # VRAM audit before training
    print("--- VRAM AUDIT (before training) ---")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print("------------------------------------")

    print("[msf_v2] Training...")
    try:
        trainer.train()
    finally:
        # Save even if interrupted
        status_writer.emit({"phase":"train","state":"saving","condition":"msf_v2",
                            "domain":args.domain,"step":args.max_steps,"total_steps":args.max_steps})
        for h in hooks:
            h.remove()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        # Cleanup
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    progress = ProgressWriter(out_dir / args.progress_file)
    progress.record({"domain": args.domain, "max_steps": args.max_steps, "output": str(out_dir)})

    if args.record_grad_norms and norm_recorders:
        print("\n[msf_v2] First-step gradient norms:")
        for (layer, proj), rec in sorted(norm_recorders.items()):
            if rec.core_norm is not None:
                print(f"  L{layer:2d} {proj:4s} core={rec.core_norm:.4f} shell={rec.shell_norm:.4f}")

    print(f"\n[msf_v2] Done. Model saved to {out_dir}")
    status_writer.emit({"phase":"train","state":"done","condition":"msf_v2",
                        "domain":args.domain,"step":args.max_steps,"total_steps":args.max_steps})

if __name__ == "__main__":
    main()