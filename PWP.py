"""
PWP - Block-Diagonal Neuron-Group Partition Experiment (v2)
============================================================
v1 failure: elementwise hash gate isolates gradients but not activations.
Gemini proof (2026-04-17): block-diagonal neuron-group partitioning gives
zero gradient flow between domains by proof. Activation contamination from
shared fc1 forward pass is a dead-end (no gradient writes follow).

v2 fix: neuron-group partitioning end-to-end.
  fc1    (H_total x 784)   domain d owns rows d*H_D:(d+1)*H_D
  fc2    (H_total x H_total)  domain d owns block [s:e, s:e], enforced by slicing
  heads  private (CLASSES x H_D) output head per domain, no cross-domain path

The forward pass slices explicitly to the active domain's neuron group.
Slicing isolates gradients at fc1 and fc2 by PyTorch autograd construction.
Gradient gate is an additional safety net, not the primary mechanism.

Capacity note (Gemini):
  H_D=128, D=2: hidden params per domain = 128^2 = 16K vs 256^2 = 65K full.
  Quadratic penalty in hidden layers. Linear at input. Irrelevant at MNIST scale.

Setup:
  H_D = 128 per domain, H_total = 256 (matches baseline model parameter budget)
  Task A: MNIST digits 0-4, local labels 0-4
  Task B: MNIST digits 5-9, local labels 0-4
  Baseline:    sequential train on same domain slot (dom 0 both phases). Expects forgetting.
  Block gate:  sequential train on separate domain slots (dom 0 then dom 1). Tests retention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ----- config -----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 3
BATCH = 128
LR = 1e-3
SEED = 42
H_D = 128
NUM_DOMAINS = 2
CLASSES_PER_DOMAIN = 5

torch.manual_seed(SEED)
print(f"device: {DEVICE}  H_D={H_D}  H_total={H_D*NUM_DOMAINS}")


# ----- model -----
class BlockMLP(nn.Module):
    """
    2-hidden-layer MLP with neuron-group domain partitioning.

    fc1: output neurons are partitioned by domain (rows).
    fc2: weight matrix has block-diagonal structure, enforced by explicit slicing.
    heads: one private (CLASSES_PER_DOMAIN x H_D) head per domain.

    Forward pass takes domain_id and routes through that domain's sub-network only.
    """
    def __init__(self, h_d=H_D, num_domains=NUM_DOMAINS, classes_per_domain=CLASSES_PER_DOMAIN):
        super().__init__()
        self.h_d = h_d
        self.h_total = h_d * num_domains
        self.fc1 = nn.Linear(784, self.h_total)
        self.fc2 = nn.Linear(self.h_total, self.h_total)
        self.heads = nn.ModuleList([
            nn.Linear(h_d, classes_per_domain) for _ in range(num_domains)
        ])

    def forward(self, x, domain_id):
        s, e = domain_id * self.h_d, (domain_id + 1) * self.h_d
        x = x.view(x.size(0), -1)
        # fc1: only the domain's output neurons participate.
        # Gradient of fc1(x)[:, s:e] flows only to fc1.weight[s:e, :].
        h1 = F.relu(self.fc1(x)[:, s:e])                   # (B, H_D)
        # fc2: explicitly use the diagonal block for this domain.
        # Gradient flows only to fc2.weight[s:e, s:e] and fc2.bias[s:e].
        w2 = self.fc2.weight[s:e, s:e]
        b2 = self.fc2.bias[s:e]
        h2 = F.relu(F.linear(h1, w2, b2))                  # (B, H_D)
        # private head: fully isolated by ModuleList, no cross-domain path.
        return self.heads[domain_id](h2)                    # (B, CLASSES)


# ----- gradient masks (safety net on top of slicing) -----
def build_neuron_masks(model):
    masks = []
    for d in range(model.num_domains):
        s, e = d * model.h_d, (d + 1) * model.h_d
        m = {}
        w1 = torch.zeros_like(model.fc1.weight); w1[s:e, :] = 1.0
        m['fc1.weight'] = w1
        b1 = torch.zeros_like(model.fc1.bias); b1[s:e] = 1.0
        m['fc1.bias'] = b1
        w2 = torch.zeros_like(model.fc2.weight); w2[s:e, s:e] = 1.0
        m['fc2.weight'] = w2
        b2 = torch.zeros_like(model.fc2.bias); b2[s:e] = 1.0
        m['fc2.bias'] = b2
        masks.append(m)
    return masks


def apply_gate(model, mask_dict):
    for name, p in model.named_parameters():
        if p.grad is not None and name in mask_dict:
            p.grad.mul_(mask_dict[name])


# ----- data -----
class RemappedSubset(Dataset):
    """MNIST Subset with labels remapped to local 0-based."""
    def __init__(self, base_ds, digit_classes):
        targets = base_ds.targets.tolist()
        class_set = set(digit_classes)
        self.offset = min(digit_classes)
        self.indices = [i for i, y in enumerate(targets) if y in class_set]
        self.base = base_ds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y - self.offset


def get_loaders(digit_classes, train=True):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST('./data', train=train, download=True, transform=tfm)
    return DataLoader(RemappedSubset(ds, digit_classes), batch_size=BATCH,
                      shuffle=train, num_workers=2, pin_memory=True)


# ----- train / eval -----
def train_phase(model, loader, optimizer, domain_id, mask=None, epochs=EPOCHS, tag=""):
    model.train()
    for ep in range(epochs):
        total = correct = loss_sum = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, domain_id)
            loss = F.cross_entropy(out, y)
            loss.backward()
            if mask is not None:
                apply_gate(model, mask)
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"    {tag} ep {ep+1}/{epochs}  loss {loss_sum/total:.4f}  acc {correct/total:.3f}")


@torch.no_grad()
def evaluate(model, loader, domain_id):
    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x, domain_id).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


# ----- experiment -----
def run(condition):
    print(f"\n=== Condition: {condition} ===")
    torch.manual_seed(SEED)
    model = BlockMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if condition == 'block_gate':
        masks = build_neuron_masks(model)
        mask_A, mask_B = masks[0], masks[1]
        dom_B = 1
    else:  # baseline: same slot for both phases, no gate
        mask_A = mask_B = None
        dom_B = 0

    dom_A = 0
    A_train = get_loaders([0, 1, 2, 3, 4], train=True)
    A_test = get_loaders([0, 1, 2, 3, 4], train=False)
    B_train = get_loaders([5, 6, 7, 8, 9], train=True)
    B_test = get_loaders([5, 6, 7, 8, 9], train=False)

    print("  [Phase A] train on 0-4")
    train_phase(model, A_train, optimizer, dom_A, mask=mask_A, tag="A")
    acc_A_post_A = evaluate(model, A_test, dom_A)
    print(f"  acc 0-4 after phase A: {acc_A_post_A:.4f}")

    print("  [Phase B] train on 5-9")
    train_phase(model, B_train, optimizer, dom_B, mask=mask_B, tag="B")
    acc_A_post_B = evaluate(model, A_test, dom_A)
    acc_B_post_B = evaluate(model, B_test, dom_B)
    print(f"  acc 0-4 after phase B: {acc_A_post_B:.4f}  <-- retention")
    print(f"  acc 5-9 after phase B: {acc_B_post_B:.4f}")

    return dict(condition=condition,
                acc_A_post_A=acc_A_post_A,
                acc_A_post_B=acc_A_post_B,
                acc_B_post_B=acc_B_post_B)


if __name__ == '__main__':
    results = [run('baseline'), run('block_gate')]

    print("\n=== Summary ===")
    hdr = f"{'condition':<12} {'A_end_A':>8} {'A_end_B':>8} {'B_end_B':>8} {'delta':>10}"
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        d = r['acc_A_post_B'] - r['acc_A_post_A']
        print(f"{r['condition']:<12} {r['acc_A_post_A']:>8.4f} {r['acc_A_post_B']:>8.4f} "
              f"{r['acc_B_post_B']:>8.4f} {d:>10.4f}")

    print()
    bg = next(r for r in results if r['condition'] == 'block_gate')
    bl = next(r for r in results if r['condition'] == 'baseline')
    if bg['acc_A_post_B'] > bl['acc_A_post_B'] + 0.5:
        print("RESULT: block_gate retained domain A. Neuron-group isolation holds.")
    elif abs(bg['acc_A_post_B'] - bl['acc_A_post_B']) < 0.05:
        print("RESULT: block_gate matched baseline forgetting. Architecture still fails.")
    else:
        print("RESULT: partial retention. Investigate shared fc2 bias or head interference.")