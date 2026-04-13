"""
Potential Well Project — Hash Bin Ablation Study (vectorized)
=============================================================
Same experiment as before, but bin averaging is fully vectorized.
No Python loops over individual parameters during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

SEED       = 42
EPOCHS     = 10
BATCH_SIZE = 256
LR         = 1e-3
BIN_SIZE   = 160
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "./data"

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

# ── Vectorized hash bin setup ─────────────────────────────────────────────────

def build_bin_index(model, bin_size, seed, device):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gen   = torch.Generator()
    gen.manual_seed(seed)
    perm    = torch.randperm(total, generator=gen)
    bin_ids = torch.empty(total, dtype=torch.long)
    bin_ids[perm] = torch.arange(total) // bin_size
    n_bins = int(bin_ids.max().item()) + 1
    print(f"[hash-bins] {n_bins} bins, {total} total parameters, "
          f"avg {total / n_bins:.1f} per bin")
    return bin_ids.to(device), n_bins


def apply_hash_gradient_averaging(model, bin_ids, n_bins):
    # 1. Collect all gradients into one flat tensor
    grads = torch.cat([
        p.grad.view(-1)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ])
    # 2. Per-bin mean via scatter_reduce
    sums = torch.zeros(n_bins, device=grads.device).scatter_reduce_(
               0, bin_ids, grads, reduce="mean", include_self=False)
    # 3. Broadcast bin means back to every position
    averaged = sums[bin_ids]
    # 4. Write back
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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader

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

# ── Full run ──────────────────────────────────────────────────────────────────

def run(use_hashing, train_loader, test_loader):
    torch.manual_seed(SEED)
    model     = MLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    bin_ids, n_bins = (build_bin_index(model, BIN_SIZE, SEED, DEVICE)
                       if use_hashing else (None, None))
    train_losses, test_losses, test_accs = [], [], []
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, loader=train_loader, optimizer=optimizer,
                              criterion=criterion, bin_ids=bin_ids, n_bins=n_bins)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)
        tag = "HASHED  " if use_hashing else "BASELINE"
        print(f"[{tag}] Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  test_loss={te_loss:.4f}  "
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
    hs_tr, hs_te, hs_acc = run(True, train_loader, test_loader)

    delta_acc = bl_acc[-1] - hs_acc[-1]
    if delta_acc < 0.02:
        verdict = "PASS — Hash bin noise does not meaningfully degrade convergence. Architecture proceeds."
    elif delta_acc < 0.05:
        verdict = "MARGINAL — Small degradation. Investigate bin size or coherence grouping."
    else:
        verdict = "FAIL — Hash bin noise is lethal. Bin design must be revised."

    report = "\n".join([
        "=" * 60,
        "Potential Well — Hash Bin Ablation Results",
        "=" * 60,
        f"Epochs:    {EPOCHS}",
        f"Bin size:  {BIN_SIZE} parameters",
        f"Device:    {DEVICE}",
        "",
        f"Baseline  final test acc:  {bl_acc[-1]*100:.2f}%",
        f"Hashed    final test acc:  {hs_acc[-1]*100:.2f}%",
        f"Delta:                     {delta_acc*100:+.2f}pp",
        "",
        f"Baseline  final test loss: {bl_te[-1]:.4f}",
        f"Hashed    final test loss: {hs_te[-1]:.4f}",
        "",
        "Verdict:", verdict,
    ])
    print("\n" + report)
    with open("ablation_results.txt", "w") as f:
        f.write(report)

    epochs = list(range(1, EPOCHS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, bl_te,  label="Baseline", color="#4C9BE8")
    axes[0].plot(epochs, hs_te,  label="Hashed",   color="#E8844C", linestyle="--")
    axes[0].set_title("Test Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [a*100 for a in bl_acc], label="Baseline", color="#4C9BE8")
    axes[1].plot(epochs, [a*100 for a in hs_acc], label="Hashed",   color="#E8844C", linestyle="--")
    axes[1].set_title("Test Accuracy (%)"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Hash Bin Ablation (bin_size={BIN_SIZE})", fontsize=13)
    plt.tight_layout()
    plt.savefig("ablation_results.png", dpi=150)
    print("Plot saved: ablation_results.png")

if __name__ == "__main__":
    main()
