"""
Proteus — Evaluation Harness
Loads a checkpoint and scores it against all 4 domain eval sets.
Reports per-domain perplexity. This is the retention measurement instrument.

Usage:
  # Score base model (establish baselines before any fine-tuning)
  python eval.py --checkpoint google/gemma-4-E4B-it --label baseline

  # Score after training domain A
  python eval.py --checkpoint checkpoints/proteus/medical --label proteus_after_medical

  # Score all checkpoints in a sweep
  python eval.py --checkpoint checkpoints/proteus/medical --label proteus_after_medical
  python eval.py --checkpoint checkpoints/proteus/legal   --label proteus_after_legal
  ... etc

Results are appended to results/eval_log.jsonl for later analysis.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
MODEL_ID    = "google/gemma-4-E4B-it"
MAX_LENGTH  = 512
DOMAINS     = ["medical", "legal", "code", "multilingual"]

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


def compute_perplexity(model, tokenizer, records: list[dict], device: str) -> float:
    """
    Compute mean perplexity over response tokens only (same masking as training).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for row in records:
            prefix, response = format_prompt_parts(row)

            prefix_ids = tokenizer(
                prefix,
                truncation=False,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids

            full_ids = tokenizer(
                prefix + response,
                truncation=True,
                max_length=MAX_LENGTH,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids

            full_ids = full_ids.to(device)

            n_prefix = min(prefix_ids.shape[1], full_ids.shape[1] - 1)
            labels   = full_ids.clone()
            labels[:, :n_prefix] = -100   # mask prompt

            n_response_tokens = (labels != -100).sum().item()
            if n_response_tokens == 0:
                continue

            outputs = model(input_ids=full_ids, labels=labels)
            # outputs.loss is mean NLL over unmasked tokens
            total_nll    += outputs.loss.item() * n_response_tokens
            total_tokens += n_response_tokens

    if total_tokens == 0:
        return float("inf")

    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


def apply_custom_lora(model, checkpoint_path: str):
    """
    Re-registers custom hook-based LoRA weights saved by train.py.
    train.py registers lora_A/lora_B as named parameters on each projection
    and saves them into model.safetensors via trainer.save_model().
    AutoModelForCausalLM.from_pretrained drops them as UNEXPECTED because
    the base architecture has no slots for them. This function:
      1. Loads the safetensors to find all lora_A/lora_B tensors
      2. Re-registers them as Parameters on the projection modules
      3. Re-registers the forward hooks that apply the LoRA delta
    """
    from safetensors.torch import load_file
    from pathlib import Path as _Path

    sf_path = _Path(checkpoint_path) / "model.safetensors"
    if not sf_path.exists():
        print("[lora] No model.safetensors found — skipping LoRA apply.")
        return []

    state = load_file(str(sf_path))
    lora_keys = {k: v for k, v in state.items() if "lora_A" in k or "lora_B" in k}
    if not lora_keys:
        print("[lora] No lora_A/lora_B keys in checkpoint — skipping LoRA apply.")
        return []

    r       = next(v for k, v in lora_keys.items() if "lora_A" in k).shape[0]
    alpha   = r * 2   # matches train.py: r=16, alpha=32 → scaling=2.0
    scaling = alpha / r

    hooks  = []
    layers = model.model.language_model.layers
    _dev   = next(model.parameters()).device

    for layer_idx, layer in enumerate(layers):
        for proj_name in ("q_proj", "v_proj"):
            if not hasattr(layer.self_attn, proj_name):
                continue
            key_A = f"model.language_model.layers.{layer_idx}.self_attn.{proj_name}.lora_A"
            key_B = f"model.language_model.layers.{layer_idx}.self_attn.{proj_name}.lora_B"
            if key_A not in lora_keys or key_B not in lora_keys:
                continue

            proj   = getattr(layer.self_attn, proj_name)
            lora_A = torch.nn.Parameter(lora_keys[key_A].to(_dev), requires_grad=False)
            lora_B = torch.nn.Parameter(lora_keys[key_B].to(_dev), requires_grad=False)
            proj.register_parameter("lora_A", lora_A)
            proj.register_parameter("lora_B", lora_B)

            def make_hook(A, B):
                def hook(module, input, output):
                    x = input[0]
                    delta = torch.nn.functional.linear(x, B @ A) * scaling
                    return output + delta
                return hook

            hooks.append(proj.register_forward_hook(make_hook(lora_A, lora_B)))

    print(f"[lora] Applied {len(hooks)} LoRA hooks from checkpoint.")
    return hooks


def load_eval_records(domain: str, n_samples: int = 500) -> list[dict]:
    path = DATA_DIR / domain / "eval.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records[:n_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="HuggingFace model ID or local checkpoint path.")
    parser.add_argument("--label",      required=True,
                        help="Human-readable label for this eval run, e.g. 'proteus_after_medical'.")
    parser.add_argument("--n_samples",  type=int, default=200,
                        help="Eval samples per domain (default 200 to keep eval fast).")
    parser.add_argument("--domains",    nargs="+", default=DOMAINS,
                        choices=DOMAINS,
                        help="Subset of domains to evaluate (default: all 4).")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== Proteus Eval ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Label:      {args.label}")
    print(f"  Domains:    {args.domains}")
    print(f"  Samples:    {args.n_samples} per domain")
    print(f"  Device:     {device}\n")

    from pathlib import Path as _Path
    import json as _json

    _adapter_cfg = _Path(args.checkpoint) / "adapter_config.json"
    _base_model_id = args.checkpoint
    if _adapter_cfg.exists():
        _base_model_id = _json.loads(_adapter_cfg.read_text()).get("base_model_name_or_path", MODEL_ID)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(_base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    if _adapter_cfg.exists():
        # LoRA / PEFT checkpoint — load base model then apply adapter
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            _base_model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, args.checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        # Custom hook-based LoRA (train.py fast path) — re-wire if lora keys present
        apply_custom_lora(model, args.checkpoint)

    results = {"label": args.label, "checkpoint": str(args.checkpoint), "domains": {}}

    for domain in args.domains:
        print(f"Evaluating {domain}...")
        records = load_eval_records(domain, n_samples=args.n_samples)
        ppl = compute_perplexity(model, tokenizer, records, device)
        results["domains"][domain] = round(ppl, 4)
        print(f"  [{domain}] perplexity = {ppl:.4f}")

    # Append to log
    log_path = RESULTS_DIR / "eval_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")

    print(f"\nResults appended to {log_path}")
    print("\nSummary:")
    for domain, ppl in results["domains"].items():
        print(f"  {domain:15s}  ppl = {ppl:.4f}")


if __name__ == "__main__":
    main()
