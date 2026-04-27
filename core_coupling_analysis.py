"""
core_coupling_analysis.py
Quantify how much each layer's core depends on the outer shell.

For each layer, we compare the core output when running at:
  - full width (core + shell)
  - core‑only width (shell excluded, using elastic width mechanics)

A high coupling score means the core output changes strongly when the
shell is removed → Tandem core → freezing the shell will hurt.
A low score means Lone core → safe to freeze the shell.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E4B-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORE_HIDDEN = 1536
CORE_INTERMEDIATE = 6144

print("Loading model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Calibration texts – a few diverse sentences
texts = [
    "The patient presents with acute abdominal pain and a history of diabetes.",
    "Under the terms of the contract, the licensee agrees to indemnify the licensor.",
    "import numpy as np\n\ndef quicksort(arr):",
    "The capital of France is Paris.",
]
enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
input_ids = enc["input_ids"].to(DEVICE)

# --------------------------------
# Hook infrastructure
# --------------------------------
layers = model.model.language_model.layers
coupling_scores = []

def full_forward():
    """Run forward pass normally (full width). Returns list of core outputs per layer."""
    core_outputs = []
    def hook_fn(layer_idx):
        def hook(module, input, output):
            # For gate_proj & up_proj: output shape [B, L, 10240]
            # The core part is the first CORE_INTERMEDIATE dimensions.
            core_out = output[..., :CORE_INTERMEDIATE].clone().detach()
            core_outputs.append((layer_idx, core_out))
        return hook

    hooks = []
    for i, layer in enumerate(layers):
        # Attach hook to gate_proj because it's the first FFN projection
        h = layer.mlp.gate_proj.register_forward_hook(hook_fn(i))
        hooks.append(h)

    with torch.no_grad():
        _ = model(input_ids)

    for h in hooks:
        h.remove()
    return core_outputs

def core_only_forward():
    """
    Run forward pass with outer shell removed.
    For gate_proj, we slice the weight matrix to use only the core part,
    then compute activations manually. This simulates the elastic inference
    at the smallest width.
    """
    core_outputs = []
    hooks = []

    def make_hook(layer_idx, proj):
        def hook(module, input, output):
            # We will replace the output manually in a post‑hook
            pass
        return hook

    with torch.no_grad():
        # We'll run the model normally but intercept the FFN computations.
        # Instead of hacking the weights, we patch the forward method of MLP
        # temporarily to use only core weights.
        original_forwards = {}
        for i, layer in enumerate(layers):
            mlp = layer.mlp
            original_forwards[i] = mlp.forward
            def core_mlp_forward(hidden_states, layer_idx=i):
                # Compute core-only gate_proj output
                gate_core = F.linear(hidden_states, mlp.gate_proj.weight[:CORE_INTERMEDIATE, :CORE_HIDDEN])
                # Similarly up_proj core (we'll just use gate for coupling)
                up_core = F.linear(hidden_states, mlp.up_proj.weight[:CORE_INTERMEDIATE, :CORE_HIDDEN])
                # down_proj uses core output (first CORE_HIDDEN intermediate)
                # but we only need gate_proj core output for the coupling metric
                # We still need to produce a full output to continue the forward.
                # For simplicity, we'll compute the full MLP but record the core gate.
                # Since we can't easily change the whole forward, we'll instead
                # record what the core output would be without the shell.
                # A cleaner way: compute gate_proj by itself outside the model.
                pass
            # This approach gets complicated. Instead, we can simply compute
            # the core part of gate_proj manually after a full forward pass.
            # But we already have the full forward. Let's do a different trick:
            # Run a forward pass where we mask the outer shell weights to zero,
            # then compute core output as before.
        # Since temporarily patching all layers is messy, let's use a post‑hook
        # on the full forward again, but this time compute the *expected* core
        # output from a modified weight matrix that only has core parameters.
        # Actually, we can re‑run the forward with a temporary weight swap.
        saved_weights = {}
        for i, layer in enumerate(layers):
            gate_w = layer.mlp.gate_proj.weight.data.clone()
            saved_weights[i] = gate_w
            # Zero out the outer part
            layer.mlp.gate_proj.weight.data[CORE_INTERMEDIATE:, :] = 0.0
            layer.mlp.gate_proj.weight.data[:, CORE_HIDDEN:] = 0.0

        # Now run forward and capture core outputs from hook
        hooks = []
        core_outputs = []
        def hook_fn(layer_idx):
            def hook(module, input, output):
                core_out = output[..., :CORE_INTERMEDIATE].clone().detach()
                core_outputs.append((layer_idx, core_out))
            return hook
        for i, layer in enumerate(layers):
            h = layer.mlp.gate_proj.register_forward_hook(hook_fn(i))
            hooks.append(h)

        _ = model(input_ids)

        for h in hooks:
            h.remove()
        # Restore weights
        for i, layer in enumerate(layers):
            layer.mlp.gate_proj.weight.data = saved_weights[i]

    return core_outputs

print("Computing full‑width core outputs ...")
full_cores = full_forward()  # list of (layer_idx, tensor)

print("Computing core‑only (masked shell) core outputs ...")
core_only_cores = core_only_forward()  # list of (layer_idx, tensor)

# Now compute coupling: cosine similarity between full and core‑only gate_proj core output
print("\nLayer‑wise coupling scores (cosine similarity between full‑width and core‑only core activations):")
scores = []
for (i_full, full_out), (i_core, core_only_out) in zip(full_cores, core_only_cores):
    # Flatten batch and sequence dimensions
    a = full_out.view(-1, full_out.shape[-1])
    b = core_only_out.view(-1, core_only_out.shape[-1])
    # Cosine similarity per token, then average
    sim = F.cosine_similarity(a, b, dim=-1).mean().item()
    scores.append((i_full, sim))
    print(f"  Layer {i_full:2d}: coupling = {sim:.4f}")

# Final summary
print("\nConclusion: layers with low cosine similarity (high coupling) depend strongly on the shell.")
print("A uniform core freeze would harm these layers the most.")
print("MSF v2 should freeze proportionally to (1 - coupling) or use a mask that preserves shell for high‑coupling layers.")