"""
Potential Well Project — Bin Size Sweep
========================================
Runs baseline once, then hashed at each bin size in BIN_SIZES.
Saves a timestamped result file per run and a combined summary + plot.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ── Config ────────────────────────────────────────────────────────────────────

SEED      = 42
EPOCHS    = 20
BATCH_SIZE = 256
LR        = 1e-3
BIN_SIZES = [16, 32, 64, 160, 320]
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR  = "./data"
OUT_DIR   = "./sweep_results"
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Vectorized bin setup ──────────────────────────────────────────────────────

def build_bin_index(model, bin_size, seed, device):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gen   = torch.Generator()
    gen.manual_seed(seed)
    perm    = torch.randperm(total, generator=gen)
    bin_ids = torch.empty(total, dtype=torch.long)
    bin_ids[perm] = torch.arange(total) // bin_size
    n_bins = int(bin_ids.max().item()) + 1
    print(f"  [hash-bins] {n_bins} bins, avg {total/n_bins:.1f} params/bin")
    return bin_ids.to(device), n_bins

def apply_hash_gradient_averaging(model, bin_ids, n_bins):
    grads = torch.cat([
        p.grad.view(-1)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ])
    sums = torch.zeros(n_bins, device=grads.device).scatter_reduce_(
               0, bin_ids, grads, reduce="mean", include_self=False)
    averaged = sums[bin_ids]
    offset = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            n = p.numel()
            p.grad.view(-1).copy_(averaged[offset : offset + n])
            offset += n

# ── Data ──────────────────────────────────────────────────────────────────────

def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test_ds,  batch_size=512,        shuffle=False, num_workers=2, pin_memory=True),
    )

# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, bin_ids=None, n_bins=None):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if bin_ids is not None:
            apply_hash_gradient_averaging(model, bin_ids, n_bins)
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

# ── Single run ────────────────────────────────────────────────────────────────

def run(label, bin_size, train_loader, test_loader):
    print(f"\n── {label} {'(bin_size=' + str(bin_size) + ')' if bin_size else ''} ──")
    torch.manual_seed(SEED)
    model     = MLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    bin_ids, n_bins = (build_bin_index(model, bin_size, SEED, DEVICE)
                       if bin_size else (None, None))

    test_accs, test_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion, bin_ids, n_bins)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion)
        test_accs.append(te_acc)
        test_losses.append(te_loss)
        print(f"  Epoch {epoch:2d}/{EPOCHS}  acc={te_acc*100:.2f}%  loss={te_loss:.4f}")

    # Save per-run file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"baseline_{ts}" if not bin_size else f"bs{bin_size}_{ts}"
    with open(os.path.join(OUT_DIR, f"{fname}.txt"), "w") as f:
        f.write(f"Label: {label}\nBin size: {bin_size}\nEpochs: {EPOCHS}\n")
        f.write(f"Final acc: {test_accs[-1]*100:.2f}%\nFinal loss: {test_losses[-1]:.4f}\n")

    return test_accs, test_losses

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = get_loaders()

    results = {}
    bl_acc, bl_loss = run("BASELINE", None, train_loader, test_loader)
    results["Baseline"] = (None, bl_acc, bl_loss)

    for bs in BIN_SIZES:
        acc, loss = run(f"HASHED bs={bs}", bs, train_loader, test_loader)
        results[f"bs={bs}"] = (bs, acc, loss)

    # ── Summary table ─────────────────────────────────────────────────────────
    bl_final = bl_acc[-1] * 100
    print("\n" + "=" * 55)
    print(f"{'Label':<12} {'Bin size':>9} {'Final acc':>10} {'Delta':>8}")
    print("=" * 55)
    print(f"{'Baseline':<12} {'—':>9} {bl_final:>9.2f}%  {'—':>7}")
    for label, (bs, acc, _) in results.items():
        if bs is None:
            continue
        delta = bl_final - acc[-1] * 100
        print(f"{label:<12} {bs:>9} {acc[-1]*100:>9.2f}%  {delta:>+7.2f}pp")
    print("=" * 55)

    # Save summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"sweep_summary_{ts}.txt"), "w") as f:
        f.write(f"Bin Size Sweep — {ts}\nEpochs: {EPOCHS}\nDevice: {DEVICE}\n\n")
        f.write(f"{'Label':<12} {'Bin size':>9} {'Final acc':>10} {'Delta':>8}\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'Baseline':<12} {'—':>9} {bl_final:>9.2f}%  {'—':>7}\n")
        for label, (bs, acc, _) in results.items():
            if bs is None:
                continue
            delta = bl_final - acc[-1] * 100
            f.write(f"{label:<12} {bs:>9} {acc[-1]*100:>9.2f}%  {delta:>+7.2f}pp\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    epochs = list(range(1, EPOCHS + 1))
    colors = ["#4C9BE8", "#E8844C", "#4CE87A", "#E84C9B", "#9B4CE8", "#E8D44C"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (label, (_, acc, loss)) in enumerate(results.items()):
        ls = "-" if label == "Baseline" else "--"
        axes[0].plot(epochs, loss,             label=label, color=colors[i], linestyle=ls)
        axes[1].plot(epochs, [a*100 for a in acc], label=label, color=colors[i], linestyle=ls)

    for ax, title, ylabel in zip(axes, ["Test Loss", "Test Accuracy (%)"], ["Loss", "Accuracy"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Hash Bin Sweep — Bin Size vs. Convergence", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"sweep_plot_{ts}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")

if __name__ == "__main__":
    main()
