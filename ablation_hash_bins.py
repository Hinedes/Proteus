"""
Potential Well Project — Hash Bin Ablation Study
=================================================
Question: Does averaging gradients within random hash buckets of ~160 weights
          degrade convergence enough to disqualify the cluster-level rigid physics?

Two runs back to back:
  1. Baseline  — normal gradient descent, no interference
  2. Hashed    — gradients averaged within hash bins before each optimizer step

Same model, same seed, same data. Compare final accuracy and loss curves.

Usage:
    python ablation_hash_bins.py

Outputs:
    ablation_results.png   — loss + accuracy curves, both runs overlaid
    ablation_results.txt   — final numbers for logging
"""

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

SEED          = 42
EPOCHS        = 10
BATCH_SIZE    = 256
LR            = 1e-3
BIN_SIZE      = 160          # parameters per hash bucket
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR      = "./data"

# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)

# ── Hash bin construction ──────────────────────────────────────────────────────

def build_bins(model: nn.Module, bin_size: int, seed: int):
    """
    Assigns every trainable scalar weight to a random bin.
    Returns a list of lists of (param_tensor, flat_index) tuples.
    Bins have at most bin_size elements; the last bin may be smaller.
    """
    rng = random.Random(seed)

    # Collect all (param, flat_idx) pairs across the model
    all_refs = []
    for param in model.parameters():
        if param.requires_grad:
            for idx in range(param.numel()):
                all_refs.append((param, idx))

    rng.shuffle(all_refs)

    bins = []
    for start in range(0, len(all_refs), bin_size):
        bins.append(all_refs[start : start + bin_size])

    total = sum(len(b) for b in bins)
    print(f"[hash-bins] {len(bins)} bins, {total} total parameters, "
          f"avg {total / len(bins):.1f} per bin")
    return bins


def apply_hash_gradient_averaging(bins):
    """
    For each bin, read the current .grad values, average them,
    and write the average back to every element in the bin.
    Call this AFTER loss.backward() and BEFORE optimizer.step().
    """
    for bin_entries in bins:
        grads = []
        for param, idx in bin_entries:
            if param.grad is not None:
                grads.append(param.grad.view(-1)[idx].item())

        if not grads:
            continue

        avg = sum(grads) / len(grads)

        for param, idx in bin_entries:
            if param.grad is not None:
                param.grad.view(-1)[idx] = avg

# ── Data ──────────────────────────────────────────────────────────────────────

def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, bins=None):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if bins is not None:
            apply_hash_gradient_averaging(bins)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

# ── Full run ──────────────────────────────────────────────────────────────────

def run(use_hashing: bool, train_loader, test_loader):
    torch.manual_seed(SEED)
    model     = MLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    bins = build_bins(model, BIN_SIZE, SEED) if use_hashing else None

    train_losses, test_losses, test_accs = [], [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, bins)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        tag = "HASHED  " if use_hashing else "BASELINE"
        print(f"[{tag}] Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  "
              f"test_loss={te_loss:.4f}  "
              f"test_acc={te_acc*100:.2f}%")

    return train_losses, test_losses, test_accs

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)

    train_loader, test_loader = get_loaders()

    print("\n── BASELINE RUN ──────────────────────────────────────────────────")
    bl_tr, bl_te, bl_acc = run(False, train_loader, test_loader)

    print("\n── HASHED RUN ────────────────────────────────────────────────────")
    hs_tr, hs_te, hs_acc = run(True,  train_loader, test_loader)

    # ── Report ────────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "Potential Well — Hash Bin Ablation Results",
        "=" * 60,
        f"Epochs:    {EPOCHS}",
        f"Bin size:  {BIN_SIZE} parameters",
        f"Device:    {DEVICE}",
        "",
        f"Baseline  final test acc:  {bl_acc[-1]*100:.2f}%",
        f"Hashed    final test acc:  {hs_acc[-1]*100:.2f}%",
        f"Delta:                     {(bl_acc[-1] - hs_acc[-1])*100:+.2f}pp",
        "",
        f"Baseline  final test loss: {bl_te[-1]:.4f}",
        f"Hashed    final test loss: {hs_te[-1]:.4f}",
        "",
        "Verdict:",
    ]

    delta_acc = bl_acc[-1] - hs_acc[-1]
    if delta_acc < 0.02:
        verdict = "PASS — Hash bin noise does not meaningfully degrade convergence. Architecture proceeds."
    elif delta_acc < 0.05:
        verdict = "MARGINAL — Small degradation observed. Investigate bin size or coherence grouping."
    else:
        verdict = "FAIL — Hash bin noise is lethal to convergence. Bin design must be revised."

    report_lines.append(verdict)
    report = "\n".join(report_lines)
    print("\n" + report)

    with open("ablation_results.txt", "w") as f:
        f.write(report)

    # ── Plot ──────────────────────────────────────────────────────────────────
    epochs = list(range(1, EPOCHS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, bl_te,  label="Baseline",  color="#4C9BE8")
    axes[0].plot(epochs, hs_te,  label="Hashed",    color="#E8844C", linestyle="--")
    axes[0].set_title("Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in bl_acc], label="Baseline",  color="#4C9BE8")
    axes[1].plot(epochs, [a*100 for a in hs_acc], label="Hashed",    color="#E8844C", linestyle="--")
    axes[1].set_title("Test Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Hash Bin Ablation (bin_size={BIN_SIZE})", fontsize=13)
    plt.tight_layout()
    plt.savefig("ablation_results.png", dpi=150)
    print("Plot saved: ablation_results.png")

if __name__ == "__main__":
    main()
