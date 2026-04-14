"""
Proteus — Dataset Pipeline
Loads, normalizes, and splits all 4 domain datasets into train/eval.

Output format (all domains):
    {"instruction": str, "input": str, "output": str, "domain": str}

Domains: medical, legal, code, multilingual
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data")
SEED = 42
EVAL_SIZE = 500  # held-out eval samples per domain


def save(split_name: str, domain: str, records: list[dict]):
    domain_dir = OUT_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    path = domain_dir / f"{split_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [{domain}] {split_name}: {len(records)} records -> {path}")


def normalize(instruction: str, input_: str, output: str, domain: str) -> dict:
    return {
        "instruction": instruction.strip(),
        "input": input_.strip(),
        "output": output.strip(),
        "domain": domain,
    }


# ─────────────────────────────────────────────
# Medical: lavita/medical-qa-datasets
# Fields: input (question), output (answer)
# ─────────────────────────────────────────────
def load_medical():
    print("[medical] Loading lavita/medical-qa-datasets ...")
    ds = load_dataset("lavita/medical-qa-datasets", "all-processed", split="train")
    records = []
    for row in ds:
        q = row.get("input") or row.get("question") or ""
        a = row.get("output") or row.get("answer") or ""
        if not q or not a:
            continue
        records.append(normalize(
            instruction="Answer the following medical question accurately.",
            input_=q,
            output=a,
            domain="medical"
        ))
    train, eval_ = records[EVAL_SIZE:], records[:EVAL_SIZE]
    save("train", "medical", train)
    save("eval", "medical", eval_)


# ─────────────────────────────────────────────
# Legal: joelniklaus/legal-mc4
# Using English subset of legal-mc4, which contains web-scraped legal text.
# Fields: text (split in half: first half is input, second half is output)
# ─────────────────────────────────────────────
def load_legal():
    print("[legal] Loading isaacus/open-australian-legal-corpus ...")
    ds = load_dataset("isaacus/open-australian-legal-corpus", split="corpus", streaming=True)
    records = []
    for row in ds:
        text = row.get("text") or ""
        if len(text) < 200:
            continue
        mid = len(text) // 2
        records.append(normalize(
            instruction="Continue the following legal text.",
            input_=text[:mid].strip(),
            output=text[mid:].strip(),
            domain="legal"
        ))
        if len(records) >= 10000:
            break
    eval_size = min(EVAL_SIZE, len(records) // 5)
    train, eval_ = records[eval_size:], records[:eval_size]
    save("train", "legal", train)
    save("eval", "legal", eval_)

# ─────────────────────────────────────────────
# Code: iamtarun/python_code_instructions_18k_alpaca
# Fields: instruction, input, output
# ─────────────────────────────────────────────
def load_code():
    print("[code] Loading iamtarun/python_code_instructions_18k_alpaca ...")
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    records = []
    for row in ds:
        instr = row.get("instruction") or ""
        inp = row.get("input") or ""
        out = row.get("output") or ""
        if not instr or not out:
            continue
        records.append(normalize(
            instruction=instr,
            input_=inp,
            output=out,
            domain="code"
        ))
    train, eval_ = records[EVAL_SIZE:], records[:EVAL_SIZE]
    save("train", "code", train)
    save("eval", "code", eval_)


# ─────────────────────────────────────────────
# Multilingual: Helsinki-NLP/opus-100
# Using English to 3 other languages (Chinese, French, Arabic) translation pairs.
# Fields: translation (dict with keys "en" and target language)
# ─────────────────────────────────────────────
def load_multilingual():
    print("[multilingual] Loading Helsinki-NLP/opus-100 ...")
    target_langs = ["zh", "fr", "ar"]
    records = []
    for lang in target_langs:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", f"en-{lang}", split="train")
        except Exception as e:
            print(f"  Warning: could not load en-{lang}: {e}")
            continue
        for row in ds:
            src = row.get("translation", {}).get("en") or ""
            tgt = row.get("translation", {}).get(lang) or ""
            if not src or not tgt:
                continue
            records.append(normalize(
                instruction=f"Translate the following English sentence to {lang}.",
                input_=src,
                output=tgt,
                domain="multilingual"
            ))
    eval_size = min(EVAL_SIZE, len(records) // 5)
    train, eval_ = records[eval_size:], records[:eval_size]
    save("train", "multilingual", train)
    save("eval", "multilingual", eval_)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    load_medical()
    load_legal()
    load_code()
    load_multilingual()
    print("\nDone. Data layout:")
    for domain in ["medical", "legal", "code", "multilingual"]:
        for split in ["train", "eval"]:
            p = OUT_DIR / domain / f"{split}.jsonl"
            if p.exists():
                lines = sum(1 for _ in open(p, encoding="utf-8"))
                print(f"  data/{domain}/{split}.jsonl  ({lines} lines)")