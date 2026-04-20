"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineering Internship – Case Study
"""

import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# ── 1. PRUNABLE LINEAR LAYER ──────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Custom linear layer with learnable gate parameters.

    Forward pass:
        gates          = sigmoid(gate_scores)       in (0, 1)
        pruned_weights = weight * gates             element-wise
        output         = F.linear(x, pruned_weights, bias)

    Gradients flow through both weight and gate_scores via chain rule.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Gate scores — init at 0 so sigmoid(0)=0.5 (neutral start)
        # L1 penalty drives them negative → sigmoid → near 0 → weight pruned
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

    def forward(self, x):
        gates          = self.get_gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


# ── 2. NETWORK ────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """Feed-forward net: 3072 → 512 → 256 → 128 → 10"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            PrunableLinear(512,  256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            PrunableLinear(256,  128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            PrunableLinear(128,   10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self):
        """
        L1 penalty = mean of all gate values across all PrunableLinear layers.
        Normalized with .mean() so it stays in (0,1), same scale as cross-entropy.
        Minimizing this drives gates toward 0 (pruned connections).
        """
        all_gates = torch.cat([l.get_gates().flatten() for l in self.prunable_layers()])
        return all_gates.mean()

    def overall_sparsity(self, threshold=0.5):
        """
        Count gates below threshold as 'pruned'.
        threshold=0.5 means gate_score < 0, i.e. network actively suppresses weight.
        This is the standard, correct way to measure sparsity with sigmoid gates.
        """
        pruned = total = 0
        for l in self.prunable_layers():
            g = l.get_gates().detach()
            pruned += (g < threshold).sum().item()
            total  += g.numel()
        return pruned / total if total else 0.0

    def all_gate_values(self):
        return np.concatenate([
            l.get_gates().detach().cpu().numpy().ravel()
            for l in self.prunable_layers()
        ])


# ── 3. DATA ───────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size=256, data_dir="./data"):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_set = torchvision.datasets.CIFAR10(data_dir, True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(data_dir, False, download=True, transform=test_tf)
    return DataLoader(train_set, shuffle=True, **kw), DataLoader(test_set, shuffle=False, **kw)


# ── 4. TRAIN / EVAL ───────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, lam):
    model.train()
    c_sum = s_sum = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        clf_loss  = F.cross_entropy(model(imgs), lbls)
        spar_loss = model.sparsity_loss()
        (clf_loss + lam * spar_loss).backward()
        optimizer.step()
        n = imgs.size(0)
        c_sum += clf_loss.item()  * n
        s_sum += spar_loss.item() * n
    N = len(loader.dataset)
    return c_sum/N, s_sum/N

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        correct += (model(imgs).argmax(1) == lbls).sum().item()
        total   += lbls.size(0)
    return correct / total


# ── 5. EXPERIMENT ─────────────────────────────────────────────────────

def run_experiment(lam, train_loader, test_loader, device,
                   epochs=30, out_dir="./results"):
    os.makedirs(out_dir, exist_ok=True)
    model = SelfPruningNet().to(device)

    # Separate optimizers: normal lr for weights, high lr for gates
    weight_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    gate_params   = [p for n, p in model.named_parameters() if 'gate' in n]
    optimizer = optim.Adam([
        {'params': weight_params, 'lr': 1e-3},
        {'params': gate_params,   'lr': 5e-3},  # gates learn faster
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  lambda={lam}  |  device={device}  |  epochs={epochs}")
    print(f"{'='*60}")

    h_clf=[]; h_spar=[]; h_acc=[]

    for epoch in range(1, epochs+1):
        clf, spar = train_one_epoch(model, train_loader, optimizer, device, lam)
        acc = evaluate(model, test_loader, device)
        sp  = model.overall_sparsity()   # threshold=0.5
        scheduler.step()
        h_clf.append(clf); h_spar.append(spar); h_acc.append(acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{epochs} | CLF={clf:.4f} Spar={spar:.4f} | "
                  f"Acc={acc*100:.2f}%  Sparsity={sp*100:.1f}%")

    final_acc = evaluate(model, test_loader, device)
    final_sp  = model.overall_sparsity()
    gates     = model.all_gate_values()
    print(f"\n  ✓ Final Accuracy : {final_acc*100:.2f}%")
    print(f"  ✓ Final Sparsity : {final_sp*100:.2f}%  (gates < 0.5)")

    # ── Gate value histogram ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(gates, bins=80, color="#2563EB", edgecolor="none", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2,
               label="Pruning threshold (0.5)")
    ax.set_xlabel("Gate Value", fontsize=12)
    ax.set_ylabel("Count",      fontsize=12)
    ax.set_title(
        f"Gate Value Distribution  |  λ={lam}  |  "
        f"Sparsity={final_sp*100:.1f}%  |  Acc={final_acc*100:.2f}%",
        fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    hp = os.path.join(out_dir, f"gate_hist_lambda_{lam}.png")
    fig.savefig(hp, dpi=150); plt.close(fig)
    print(f"  ✓ Histogram saved → {hp}")

    # ── Loss & accuracy curves ───────────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, epochs+1)
    axes[0].plot(ep, h_clf,  label="CLF Loss",      color="#2563EB")
    axes[0].plot(ep, h_spar, label="Sparsity Loss", color="#F97316")
    axes[0].set_title(f"Loss Curves (λ={lam})"); axes[0].legend()
    axes[1].plot(ep, [a*100 for a in h_acc], color="#10B981")
    axes[1].set_title(f"Test Accuracy (λ={lam})"); axes[1].set_ylabel("Acc (%)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"curves_lambda_{lam}.png"), dpi=150)
    plt.close(fig2)

    return {
        "lambda":        lam,
        "test_accuracy": round(final_acc * 100, 2),
        "sparsity":      round(final_sp  * 100, 2),
    }


# ── 6. MAIN ───────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = get_cifar10_loaders()

    # Three lambda values: low / medium / high
    lambdas = [1.0, 5.0, 15.0]
    results = []
    for lam in lambdas:
        r = run_experiment(lam, train_loader, test_loader, device, epochs=30)
        results.append(r)

    print("\n\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("-"*55)
    for r in results:
        print(f"{r['lambda']:<12} {r['test_accuracy']:>14} {r['sparsity']:>14}")
    print("="*55)

    os.makedirs("./results", exist_ok=True)
    with open("./results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  ✓ Results saved → ./results/results.json")

if __name__ == "__main__":
    main()

