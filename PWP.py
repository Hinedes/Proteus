"""
PWP - Minimal Hash-Gate Experiment
==================================
Claim under test: a deterministic hash partition of parameter indices,
used as a gradient gate during sequential training, prevents catastrophic
forgetting without replay, regularization, or learned masks.

Design:
    Model   2-hidden-layer MLP (784 -> 256 -> 256 -> 10)
    Task A  MNIST digits 0-4
    Task B  MNIST digits 5-9
    Order   sequential: train A, then train B

Conditions:
    baseline    no gate. Expected: acc on A collapses after phase B.
    hash_gate   hash(param_name, flat_index) mod 2 picks domain for each
                scalar weight. Gradients outside the active domain are zeroed.

Metric: accuracy on 0-4 after phase B completes (retention).
"""

import hashlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----- config -----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS_PER_PHASE = 3
BATCH = 128
LR = 1e-3
SEED = 42
HIDDEN = 256
NUM_DOMAINS = 2

torch.manual_seed(SEED)


# ----- model -----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----- hash gate -----
def build_hash_masks(model, num_domains=NUM_DOMAINS, seed=SEED):
    """
    For each parameter tensor, produce per-domain boolean masks partitioning
    its scalar elements deterministically by md5(name, index, seed) mod N.
    Returns a list of dicts: [ {param_name: mask_tensor}, ... ] per domain.
    """
    per_domain = [{} for _ in range(num_domains)]
    for name, p in model.named_parameters():
        n = p.numel()
        assignments = torch.empty(n, dtype=torch.long)
        for i in range(n):
            h = hashlib.md5(f"{name}:{i}:{seed}".encode()).digest()
            assignments[i] = int.from_bytes(h[:4], 'big') % num_domains
        assignments = assignments.view_as(p).to(p.device)
        for d in range(num_domains):
            per_domain[d][name] = (assignments == d).float()
    return per_domain


def apply_gate(model, mask_dict):
    """Zero gradients outside the active domain mask, in-place."""
    for name, p in model.named_parameters():
        if p.grad is not None and name in mask_dict:
            p.grad.mul_(mask_dict[name])


# ----- data -----
def get_split_mnist(classes, train=True):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST('./data', train=train, download=True, transform=tfm)
    class_set = set(classes)
    targets = ds.targets.tolist()
    idx = [i for i, y in enumerate(targets) if y in class_set]
    return Subset(ds, idx)


# ----- train / eval -----
def train_phase(model, loader, optimizer, gate_mask=None, epochs=EPOCHS_PER_PHASE, tag=""):
    model.train()
    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            if gate_mask is not None:
                apply_gate(model, gate_mask)
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"    {tag} epoch {epoch+1}/{epochs}  loss {loss_sum/total:.4f}  train_acc {correct/total:.3f}")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


# ----- experiment driver -----
def run(condition):
    print(f"\n=== Condition: {condition} ===")
    torch.manual_seed(SEED)
    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if condition == 'hash_gate':
        t0 = time.time()
        masks = build_hash_masks(model)
        print(f"  built hash masks in {time.time()-t0:.2f}s")
        mask_A, mask_B = masks[0], masks[1]
        # sanity: coverage of domain A across the whole model
        total = sum(m.numel() for m in mask_A.values())
        live = sum(m.sum().item() for m in mask_A.values())
        print(f"  domain A covers {live/total:.3f} of params ({int(live)}/{total})")
    else:
        mask_A = mask_B = None

    A_train = DataLoader(get_split_mnist([0, 1, 2, 3, 4], train=True),
                         batch_size=BATCH, shuffle=True)
    A_test = DataLoader(get_split_mnist([0, 1, 2, 3, 4], train=False),
                        batch_size=BATCH)
    B_train = DataLoader(get_split_mnist([5, 6, 7, 8, 9], train=True),
                         batch_size=BATCH, shuffle=True)
    B_test = DataLoader(get_split_mnist([5, 6, 7, 8, 9], train=False),
                        batch_size=BATCH)

    print("  [Phase A] train on 0-4")
    train_phase(model, A_train, optimizer, gate_mask=mask_A, tag="A")
    acc_A_after_A = evaluate(model, A_test)
    print(f"  acc on 0-4 after phase A: {acc_A_after_A:.4f}")

    print("  [Phase B] train on 5-9")
    train_phase(model, B_train, optimizer, gate_mask=mask_B, tag="B")
    acc_A_after_B = evaluate(model, A_test)
    acc_B_after_B = evaluate(model, B_test)
    print(f"  acc on 0-4 after phase B: {acc_A_after_B:.4f}  <-- retention")
    print(f"  acc on 5-9 after phase B: {acc_B_after_B:.4f}")

    return {
        'condition': condition,
        'acc_A_after_A': acc_A_after_A,
        'acc_A_after_B': acc_A_after_B,
        'acc_B_after_B': acc_B_after_B,
    }


if __name__ == '__main__':
    print(f"device: {DEVICE}")
    results = [run('baseline'), run('hash_gate')]

    print("\n=== Summary ===")
    header = f"{'condition':<12} {'A_end_A':>8} {'A_end_B':>8} {'B_end_B':>8} {'retention_delta':>18}"
    print(header)
    print("-" * len(header))
    for r in results:
        delta = r['acc_A_after_B'] - r['acc_A_after_A']
        print(f"{r['condition']:<12} {r['acc_A_after_A']:>8.4f} "
              f"{r['acc_A_after_B']:>8.4f} {r['acc_B_after_B']:>8.4f} "
              f"{delta:>18.4f}")
    print("\nInterpretation:")
    print("  baseline retention_delta near -0.95  => catastrophic forgetting, as expected")
    print("  hash_gate retention_delta near  0.00 => hash-only claim holds on this toy")
    print("  hash_gate retention_delta similar to baseline => claim fails, Lennard-Jones was not optional")