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
        
        # 1. Extract the Core quadrant
        if ptype != "down":
            core = grad[:CORE_MID, :CORE_HIDDEN]
            outer_bottom = grad[CORE_MID:, :]
            outer_top_right = grad[:CORE_MID, CORE_HIDDEN:]
        else:
            core = grad[:CORE_HIDDEN, :CORE_MID]
            outer_bottom = grad[CORE_HIDDEN:, :]
            outer_top_right = grad[:CORE_HIDDEN, CORE_MID:]
            
        # 2. Find the maximum gradient across ALL outer regions
        outer_max = max(outer_bottom.abs().max().item(), outer_top_right.abs().max().item())
        core_max = core.abs().max().item()
        
        # 3. Assert Core is perfectly frozen (0.0) and Outer is learning (> 0.0)
        if core_max == 0.0 and outer_max > 0.0:
            passed += 1
        else:
            print(f"[FAIL] Layer {i} {name}: core_max={core_max:.2e} outer_max={outer_max:.2e}")
            failed += 1

print(f"\n{'PASSED' if failed==0 else 'FAILED'}: {passed}/{passed+failed} checks.")
for h in hooks: h.remove()
