#!/usr/bin/env python3
"""
grad_norm_probe.py
One‑shot gradient norm measurement on a single batch of Medical data.

For each FFN layer, computes:
  - core_norm  = Frobenius norm of gradients flowing into the frozen core rectangle
                 (as defined by MSF v2's per‑layer dimensions)
  - shell_norm = Frobenius norm of gradients in the remaining outer shell
  - ratio      = core_norm / shell_norm

This reveals whether Full FT’s error signal preferentially strikes the
shell (ratio < 1) – meaning the core is already in a wide basin – or
hits the core equally (ratio ~ 1).

Usage:
  python grad_norm_probe.py
  python grad_norm_probe.py --coupling_csv coupling.csv --batch_size 8
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID       = "google/gemma-4-E4B-it"
CORE_HIDDEN    = 1536
CORE_MID       = 6144
MAX_LENGTH     = 512
DATA_DIR       = Path("data")
RESPONSE_KEY   = "### Response:\n"

# Default coupling scores (same as train_msf_v2.py)
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
    if csv_path:
        import csv
        scores = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row.get("layer", row.get("Layer")))
                score = float(row.get("coupling", row.get("s_l")))
                scores[layer] = score
        return scores
    return DEFAULT_COUPLING

def per_layer_freeze_dims(coupling):
    dims = {}
    for layer, s in coupling.items():
        f = s ** 2
        m = round(f * CORE_MID)
        n = round(f * CORE_HIDDEN)
        dims[layer] = (m, n)
    return dims

def load_medical_batch(batch_size: int):
    """Load a single batch from medical train data."""
    path = DATA_DIR / "medical" / "train.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    # sample a small subset to keep it fast
    subset = random.sample(records, min(batch_size, len(records)))
    ds = Dataset.from_list(subset)
    return ds

def format_prompt(row):
    instruction = row["instruction"]
    inp = row.get("input", "").strip()
    output = row["output"]
    if inp:
        prefix = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n{RESPONSE_KEY}"
    else:
        prefix = f"### Instruction:\n{instruction}\n\n{RESPONSE_KEY}"
    return prefix, output

def build_inputs(ds, tokenizer):
    """Tokenize a small set of examples without truncation padding complexities."""
    input_ids_list = []
    labels_list = []
    for i in range(len(ds)):
        prefix, response = format_prompt({
            "instruction": ds[i]["instruction"],
            "input": ds[i].get("input", ""),
            "output": ds[i]["output"],
        })
        full_text = prefix + response
        enc = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH,
                        return_tensors="pt", add_special_tokens=True)
        prefix_ids = tokenizer(prefix, truncation=False, add_special_tokens=True)["input_ids"]
        n_prefix = min(len(prefix_ids), enc.input_ids.shape[1] - 1)
        labels = enc.input_ids.clone()
        labels[:, :n_prefix] = -100
        input_ids_list.append(enc.input_ids)
        labels_list.append(labels)

    # pad to longest in batch
    input_ids = torch.nn.functional.pad(
        torch.cat(input_ids_list, dim=0),
        (0, MAX_LENGTH - max(t.shape[1] for t in input_ids_list)),
        value=0
    )[:, :MAX_LENGTH]
    labels = torch.nn.functional.pad(
        torch.cat(labels_list, dim=0),
        (0, MAX_LENGTH - max(t.shape[1] for t in labels_list)),
        value=-100
    )[:, :MAX_LENGTH]
    return input_ids, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--coupling_csv", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()    # only need forward to compute gradients, but we'll enable grad
    # Actually we need the model in train mode so that requires_grad works for hooks? No, hook works either.
    # We'll set requires_grad on the model parameters? We'll compute loss.backward() which will require params to have requires_grad=True. The model is loaded in inference mode (model.eval()) but that just affects dropout/batchnorm. The parameters still have requires_grad=True by default. So loss.backward() works.
    # However, we need to temporarily set trainable? We'll keep as is.

    # Load coupling and dims
    coupling = load_coupling(args.coupling_csv)
    dims = per_layer_freeze_dims(coupling)

    # Get medical batch
    ds = load_medical_batch(args.batch_size)
    input_ids, labels = build_inputs(ds, tokenizer)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Hooks to capture gradients
    layers = model.model.language_model.layers
    grad_norms = []   # list of (layer, proj, core_norm, shell_norm)

    # We'll attach a hook to each weight tensor that saves the grad norms right after backward.
    # Since backward will compute the gradients, we need the hook to fire at the end of backward pass.
    # .register_hook on a tensor fires when that tensor's gradient is computed.
    handles = []
    capture = {}

    for i, layer in enumerate(layers):
        if i not in dims:
            m, n = CORE_MID, CORE_HIDDEN
        else:
            m, n = dims[i]

        for proj in ("gate", "up", "down"):
            param = getattr(layer.mlp, f"{proj}_proj").weight
            # Capture the dimensions for this layer/proj in a closure
            def make_hook(l=i, p=proj, _m=m, _n=n):
                def hook(grad):
                    # grad is a copy? Actually the hook receives the gradient tensor after it's been accumulated.
                    # We can compute its norm without modifying.
                    with torch.no_grad():
                        grad_det = grad.detach()
                        if p != "down":
                            core = grad_det[:_m, :_n]
                            shell_frags = [grad_det[_m:, :].flatten(),
                                           grad_det[:_m, _n:].flatten()]
                            shell = torch.cat(shell_frags)
                        else:
                            core = grad_det[:_n, :_m]
                            shell_frags = [grad_det[_n:, :].flatten(),
                                           grad_det[:_n, _m:].flatten()]
                            shell = torch.cat(shell_frags)
                        core_norm = core.norm().item()
                        shell_norm = shell.norm().item()
                        capture[(l, p)] = (core_norm, shell_norm)
                return hook
            h = param.register_hook(make_hook())
            handles.append(h)

    # Forward + backward
    model.train()  # enable dropout? Not needed but help ensure params have grad enabled. Usually they do.
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    print(f"Batch loss: {loss.item():.4f}")
    loss.backward()

    # Collect results
    for h in handles:
        h.remove()

    # Print table
    print("\nLayer‑wise gradient norms (first batch, medical domain):")
    print(f"{'Layer':>5}  {'Proj':>4}  {'Core norm':>10}  {'Shell norm':>10}  {'Ratio (core/shell)':>18}")
    print("-" * 60)
    for layer_idx in sorted(dims.keys()):
        for proj in ("gate", "up", "down"):
            if (layer_idx, proj) in capture:
                core_n, shell_n = capture[(layer_idx, proj)]
                ratio = core_n / shell_n if shell_n > 1e-12 else float('inf')
                print(f"{layer_idx:5d}  {proj:4s}  {core_n:10.4f}  {shell_n:10.4f}  {ratio:18.4f}")

    # Summary per layer (average across gate/up/down)
    print("\nLayer average (gate/up/down):")
    for layer_idx in sorted(dims.keys()):
        total_core = total_shell = 0.0
        count = 0
        for proj in ("gate", "up", "down"):
            if (layer_idx, proj) in capture:
                total_core += capture[(layer_idx, proj)][0]
                total_shell += capture[(layer_idx, proj)][1]
                count += 1
        if count:
            avg_core = total_core / count
            avg_shell = total_shell / count
            avg_ratio = avg_core / avg_shell if avg_shell > 1e-12 else float('inf')
            print(f"  L{layer_idx:2d}: core={avg_core:.4f}  shell={avg_shell:.4f}  ratio={avg_ratio:.4f}")

if __name__ == "__main__":
    main()