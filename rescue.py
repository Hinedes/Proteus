import torch
import gc
from transformers import AutoModelForCausalLM
import train # Imports your existing compute_fisher and save_ewc_state functions

# 1. Load the checkpoint that successfully saved
out_dir = "checkpoints/ewc_canon/medical"
print(f"Loading {out_dir}...")
model = AutoModelForCausalLM.from_pretrained(out_dir, torch_dtype=torch.bfloat16, device_map="auto")

# 2. Load the dataset (we just need a chunk to compute fisher)
print("Loading dataset...")
from datasets import load_dataset
dataset = load_dataset("lavita/medical-qa-datasets", split="train")
# Use your exact tokenization logic here. For safety, just grab 200 samples.
# (Assuming your compute_fisher handles the raw text or you tokenize it first)
# You might need to adapt this exact dataset loading line to match how train.py passes 'tokenized' to compute_fisher.
tokenized = train.prepare_dataset("medical") # Assuming you have a helper like this

# 3. Compute and save
print("Computing Fisher...")
new_fisher, new_opt = train.compute_fisher(model, tokenized, n_samples=200)
train.save_ewc_state(new_fisher, new_opt, out_dir)
print("Rescue complete! fisher.pt saved.")