# 🌊 Proteus

**Matryoshka Plasticity: Exploiting Nested Transformer Structure for Zero‑Overhead Continual Learning**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19841424.svg)](https://doi.org/10.5281/zenodo.19841424)

Activation hierarchy **is** protection hierarchy.  
MatFormer's nested structure gives a free importance ranking — freeze the Core, train the outer shells, and retain previous domains with zero overhead.

Zero Fisher matrices. Zero replay buffers. Zero added parameters.

## Paper

**Preprint:** [10.5281/zenodo.19841424](https://doi.org/10.5281/zenodo.19841424)  
*Matryoshka Plasticity: Exploiting Nested Transformer Structure for Zero‑Overhead Continual Learning*  
Amantur Saliev, 2026

### Key results (Gemma‑4‑E4B, Medical → Legal → Code)
- After two domains: MSF reduces Medical forgetting by **15.9%** vs. full fine‑tuning.
- After three domains: full fine‑tuning recovers via positive backward transfer (Write‑Lock Dilemma).
- Shell saturation sets a hard capacity limit for long domain chains.
- Attention layers must stay fully trainable (no nesting).

> *"MatFormer researchers forced the core into storing invariant data - I just froze it."*

## How to cite

> [1] A. Saliev, "Matryoshka Plasticity: Exploiting Nested Transformer Structure for Zero‑Overhead Continual Learning," Zenodo, Apr. 28, 2026. doi: 10.5281/zenodo.19841424.

## Quickstart

```bash
git clone https://github.com/Hinedes/Proteus.git
cd Proteus

# Install dependencies (adjust for your environment)
pip install transformers datasets safetensors

# Build the datasets
python pipeline.py

# Train a single domain (Medical, MSF)
python train.py --domain medical --condition proteus --out_dir checkpoints/msf --batch_size 16 --grad_accum 1 --max_steps 2000

# Evaluate
python eval.py --checkpoint checkpoints/msf/medical --label msf_after_medical --n_samples 100