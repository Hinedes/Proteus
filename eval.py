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
import logging
import math
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformer_engine").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
MODEL_ID    = "google/gemma-4-E4B-it"
MAX_LENGTH  = 512
DOMAINS     = ["medical", "legal", "code", "multilingual"]

RESPONSE_KEY = "### Response:\n"


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
            # Never let status I/O interfere with evaluation.
            pass


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


def compute_perplexity(model, tokenizer, records: list[dict], device: str, progress_cb=None) -> float:
    """
    Compute mean perplexity over response tokens only (same masking as training).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    started = time.time()
    total_records = len(records)

    with torch.no_grad():
        for i, row in enumerate(records, start=1):
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
                if progress_cb is not None:
                    elapsed = max(time.time() - started, 1e-6)
                    running_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
                    progress_cb(i, total_records, elapsed, running_ppl)
                continue

            outputs = model(input_ids=full_ids, labels=labels)
            # outputs.loss is mean NLL over unmasked tokens
            total_nll    += outputs.loss.item() * n_response_tokens
            total_tokens += n_response_tokens

            if progress_cb is not None:
                elapsed = max(time.time() - started, 1e-6)
                running_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
                progress_cb(i, total_records, elapsed, running_ppl)

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
    Handles both attention LoRA (q_proj, v_proj) and FFN LoRA
    (gate_proj, up_proj, down_proj) by routing from the key names.
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
    alpha   = r * 2   # scaling = alpha/r = 2.0 (matches train.py r=64, alpha=128)
    scaling = alpha / r

    hooks  = []
    layers = model.model.language_model.layers
    _dev   = next(model.parameters()).device

    def _register(module_obj, key_A, key_B):
        lora_A = torch.nn.Parameter(lora_keys[key_A].to(_dev), requires_grad=False)
        lora_B = torch.nn.Parameter(lora_keys[key_B].to(_dev), requires_grad=False)
        module_obj.register_parameter("lora_A", lora_A)
        module_obj.register_parameter("lora_B", lora_B)

        def make_hook(A, B):
            def hook(module, input, output):
                x = input[0]
                delta = torch.nn.functional.linear(x, B @ A) * scaling
                return output + delta
            return hook

        hooks.append(module_obj.register_forward_hook(make_hook(lora_A, lora_B)))

    for layer_idx, layer in enumerate(layers):
        # Attention LoRA (standard baseline: q_proj, v_proj)
        for proj_name in ("q_proj", "v_proj"):
            if not hasattr(layer.self_attn, proj_name):
                continue
            key_A = f"model.language_model.layers.{layer_idx}.self_attn.{proj_name}.lora_A"
            key_B = f"model.language_model.layers.{layer_idx}.self_attn.{proj_name}.lora_B"
            if key_A not in lora_keys or key_B not in lora_keys:
                continue
            _register(getattr(layer.self_attn, proj_name), key_A, key_B)

        # FFN LoRA (lora_ffn ablation: gate_proj, up_proj, down_proj)
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            if not hasattr(layer.mlp, proj_name):
                continue
            key_A = f"model.language_model.layers.{layer_idx}.mlp.{proj_name}.lora_A"
            key_B = f"model.language_model.layers.{layer_idx}.mlp.{proj_name}.lora_B"
            if key_A not in lora_keys or key_B not in lora_keys:
                continue
            _register(getattr(layer.mlp, proj_name), key_A, key_B)

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
    parser.add_argument("--status_file", type=str, default=None,
                        help="Optional JSON status file path for live progress rendering.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    status_writer = StatusWriter(args.status_file)
    overall_total = max(args.n_samples * len(args.domains), 1)
    overall_done = 0

    status_writer.emit({
        "phase": "eval",
        "state": "initializing",
        "label": args.label,
        "domain": "",
        "domain_index": 0,
        "domains_total": len(args.domains),
        "sample": 0,
        "sample_total": 0,
        "overall_step": 0,
        "overall_total": overall_total,
        "it_s": 0.0,
        "ppl_so_far": None,
    })

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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

    for domain_index, domain in enumerate(args.domains, start=1):
        print(f"Evaluating {domain}...")
        records = load_eval_records(domain, n_samples=args.n_samples)

        status_writer.emit({
            "phase": "eval",
            "state": "running",
            "label": args.label,
            "domain": domain,
            "domain_index": domain_index,
            "domains_total": len(args.domains),
            "sample": 0,
            "sample_total": len(records),
            "overall_step": overall_done,
            "overall_total": max(overall_total, overall_done + len(records)),
            "it_s": 0.0,
            "ppl_so_far": None,
        })

        def _progress(sample_idx, sample_total, elapsed_s, running_ppl):
            status_writer.emit({
                "phase": "eval",
                "state": "running",
                "label": args.label,
                "domain": domain,
                "domain_index": domain_index,
                "domains_total": len(args.domains),
                "sample": sample_idx,
                "sample_total": sample_total,
                "overall_step": overall_done + sample_idx,
                "overall_total": max(overall_total, overall_done + sample_total),
                "it_s": (sample_idx / elapsed_s) if elapsed_s > 0 else 0.0,
                "ppl_so_far": running_ppl,
            })

        ppl = compute_perplexity(model, tokenizer, records, device, progress_cb=_progress)
        results["domains"][domain] = round(ppl, 4)
        overall_done += len(records)

        status_writer.emit({
            "phase": "eval",
            "state": "domain_done",
            "label": args.label,
            "domain": domain,
            "domain_index": domain_index,
            "domains_total": len(args.domains),
            "sample": len(records),
            "sample_total": len(records),
            "overall_step": overall_done,
            "overall_total": max(overall_total, overall_done),
            "it_s": 0.0,
            "ppl_so_far": ppl,
        })
        print(f"  [{domain}] perplexity = {ppl:.4f}")

    # Append to log
    log_path = RESULTS_DIR / "eval_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")

    print(f"\nResults appended to {log_path}")
    print("\nSummary:")
    for domain, ppl in results["domains"].items():
        print(f"  {domain:15s}  ppl = {ppl:.4f}")

    status_writer.emit({
        "phase": "eval",
        "state": "done",
        "label": args.label,
        "domain": "",
        "domain_index": len(args.domains),
        "domains_total": len(args.domains),
        "sample": 0,
        "sample_total": 0,
        "overall_step": overall_done,
        "overall_total": max(overall_total, overall_done),
        "it_s": 0.0,
        "ppl_so_far": None,
    })


if __name__ == "__main__":
    main()
