import torch
import json
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# --- CONFIG ---
MODEL_PATH = "/scratch/checkpoints/ewc_seed1/legal"
DATA_PATH  = "data/legal/train.jsonl"
OUT_PATH   = Path(MODEL_PATH) / "fisher.pt"
MODEL_ID   = "google/gemma-4-E4B-it" 
N_SAMPLES  = 200
BATCH_SIZE = 2  # The VRAM safe-zone

def load_data():
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)

def format_prompt(row):
    prefix = f"### Instruction:\n{row['instruction']}\n\n"
    if row.get("input"):
        prefix += f"### Input:\n{row['input']}\n\n"
    prefix += "### Response:\n"
    return prefix, row["output"]

def tokenize(ds, tokenizer):
    def fn(batch):
        ids, labels = [], []
        for i in range(len(batch["instruction"])):
            p, r = format_prompt({k: v[i] for k, v in batch.items()})
            p_ids = tokenizer(p, add_special_tokens=True)["input_ids"]
            f_ids = tokenizer(p + r, truncation=True, max_length=512, add_special_tokens=True)["input_ids"]
            n_p = min(len(p_ids), len(f_ids)-1)
            labs = [-100]*n_p + f_ids[n_p:]
            ids.append(f_ids); labels.append(labs)
        return {"input_ids": ids, "labels": labels}
    return ds.map(fn, batched=True, remove_columns=ds.column_names)

def run():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Use bfloat16 and auto device mapping to mimic train.py
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print("Preparing dataset...")
    ds = load_data()
    tokenized = tokenize(ds, tokenizer)
    
    print(f"Computing Fisher (Samples={N_SAMPLES}, BS={BATCH_SIZE})...")
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    opt_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    indices = random.sample(range(len(tokenized)), min(N_SAMPLES, len(tokenized)))
    n_batches = 0
    
    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        samples = [tokenized[j] for j in batch_idx]
        
        # Manually pad for the small batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s["input_ids"]) for s in samples],
            batch_first=True, padding_value=0
        ).to(model.device)
        
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s["labels"]) for s in samples],
            batch_first=True, padding_value=-100
        ).to(model.device)
        
        model.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_batches += 1
        if (n_batches * BATCH_SIZE) % 20 == 0:
            print(f"  Processed {n_batches * BATCH_SIZE}/{N_SAMPLES} samples...")

    for n in fisher: fisher[n] /= n_batches
    
    print(f"Saving Fisher state to {OUT_PATH}...")
    torch.save({"fisher": fisher, "opt_params": opt_params}, OUT_PATH)
    print("Recovery Complete.")

if __name__ == "__main__":
    run()