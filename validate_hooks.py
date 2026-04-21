import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID     = "google/gemma-4-E4B-it"
CORE_HIDDEN  = 1536
CORE_MID     = 6144
DEVICE       = "cuda"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
).to(DEVICE)
model.train()
print("Model loaded.\n")

def make_hook(proj_type):
    def hook(grad):
        g = grad.clone()
        if proj_type in ("gate", "up"):
            g[:CORE_MID, :CORE_HIDDEN] = 0.0
        else:
            g[:CORE_HIDDEN, :CORE_MID] = 0.0
        return g
    return hook

hooks = []
layers = model.model.language_model.layers
for layer in layers:
    hooks.append(layer.mlp.gate_proj.weight.register_hook(make_hook("gate")))
    hooks.append(layer.mlp.up_proj.weight.register_hook(make_hook("up")))
    hooks.append(layer.mlp.down_proj.weight.register_hook(make_hook("down")))
print(f"Registered {len(hooks)} hooks on {len(layers)} layers.\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
inputs = tokenizer("The capital of France is", return_tensors="pt").to(DEVICE)
outputs = model(**inputs, labels=inputs["input_ids"])
print(f"Loss: {outputs.loss.item():.4f}")
outputs.loss.backward()
print("Backward done.\n")

passed, failed = 0, 0
for i, layer in enumerate(layers):
    for name, ptype in [("gate_proj","gate"),("up_proj","up"),("down_proj","down")]:
        grad = getattr(layer.mlp, name).weight.grad
        if grad is None:
            print(f"[FAIL] Layer {i} {name}: grad None"); failed += 1; continue
        core  = grad[:CORE_MID, :CORE_HIDDEN] if ptype != "down" else grad[:CORE_HIDDEN, :CORE_MID]
        outer = grad[CORE_MID:, :] if ptype != "down" else grad[CORE_HIDDEN:, :]
        if core.abs().max() == 0.0 and outer.abs().max() > 0.0:
            passed += 1
        else:
            print(f"[FAIL] Layer {i} {name}: core_max={core.abs().max():.2e} outer_max={outer.abs().max():.2e}")
            failed += 1

print(f"\n{'PASSED' if failed==0 else 'FAILED'}: {passed}/{passed+failed} checks.")
for h in hooks: h.remove()
