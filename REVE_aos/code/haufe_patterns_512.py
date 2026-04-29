"""Haufe activation patterns lifted into REVE's 512-D output space.

For the trained K=50 V/U-subspace probe, decompose each class's weight slice into
(1) the per-slot token weights, and (2) the CLS weight. Lift each into REVE's 512-D
output space using V_K and U_K, multiply by the matching uncentered covariance
(Sigma_tok in the V eigenbasis = diag(eig_V); Sigma_cls in the U eigenbasis =
diag(eig_U)), and reduce to:

  P_tok_class_full ∈ R^{C × P × 512}   per (class, channel, patch) 512-D pattern
  P_class_512      ∈ R^{512}           per-class summary, sum over (channel, patch)
  P_cls_class_512  ∈ R^{512}           per-class CLS contribution
  P_total_class    ∈ R^{512}           sum of the two

Visualization focuses on REVE's 512-D output space — no scalp topomaps.

Outputs:
  reve_haufe512.npz  - all patterns and W
  reve_haufe512.pdf  - 512-D pattern curves per class, per-(class, channel) heatmap,
                        per-(class, patch) heatmap, top-coord ranking per class
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import balanced_accuracy_score

FEAT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_subspace_features.npz")
V_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_main_subspace.npz")
U_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_cls_subspace.npz")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_haufe512.npz")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_haufe512.pdf")

D_EMB = 512
K = 50
N_CHAN = 64
N_PATCH = 4
N_TOK = N_CHAN * N_PATCH
N_FEAT = K + N_TOK * K
N_CLS = 4
EPOCHS = 20
BATCH = 32
LR = 1e-4
WD = 1e-2
DROPOUT = 0.1
PATIENCE = 5
SEED = 0
CLASS_NAMES = ["L", "R", "B", "F"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class SubspaceHead(nn.Module):
    def __init__(self, n_features, n_classes=N_CLS, dropout=DROPOUT):
        super().__init__()
        self.norm = RMSNorm(n_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.fc(self.drop(self.norm(x)))


def iterate(X, y, batch=BATCH, shuffle=True, rng=None):
    n = X.shape[0]; idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        sl = idx[i:i + batch]
        yield torch.from_numpy(X[sl].astype(np.float32)), torch.from_numpy(y[sl])


def evaluate(model, X, y):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in iterate(X, y, shuffle=False):
            preds.append(model(xb).argmax(-1).numpy())
    preds = np.concatenate(preds)
    return balanced_accuracy_score(y, preds), (preds == y).mean()


def train_probe(Xtr, ytr, Xva, yva):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model = SubspaceHead(N_FEAT)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, nb = 0.0, 0
        for xb, yb in iterate(Xtr, ytr, shuffle=True, rng=rng):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += loss.item(); nb += 1
        val_bal, _ = evaluate(model, Xva, yva)
        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"  ep {epoch:>2}: val_bal={val_bal:.4f} best={best_val:.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_val


def main():
    d = np.load(FEAT_NPZ)
    features = d["features"]; labels = d["labels"].astype(np.int64); subjects = d["subjects"]

    train_sids = list(range(1, 71)); val_sids = list(range(71, 90)); test_sids = list(range(90, 110))
    m_tr = np.isin(subjects, train_sids); m_va = np.isin(subjects, val_sids); m_te = np.isin(subjects, test_sids)

    print(f"train/val/test = {m_tr.sum()}/{m_va.sum()}/{m_te.sum()}", flush=True)
    print("Training V/U probe ...", flush=True)
    model, best_val = train_probe(features[m_tr], labels[m_tr], features[m_va], labels[m_va])
    test_bal, _ = evaluate(model, features[m_te], labels[m_te])
    print(f"best val={best_val:.4f}  test bal={test_bal:.4f}", flush=True)

    W = model.fc.weight.detach().numpy()              # (4, 12850)

    # Decompose
    W_cls = W[:, :K]                                   # (4, 50) U-coords
    W_tok = W[:, K:].reshape(N_CLS, N_TOK, K)          # (4, 256, 50) V-coords per slot
    W_tok = W_tok.reshape(N_CLS, N_CHAN, N_PATCH, K)   # (class, channel, patch, V-comp)

    # Eigenvalues and eigenvectors
    eig_V = np.load(V_NPZ)["eigvals"][:K].astype(np.float32)
    eig_U = np.load(U_NPZ)["eigvals"][:K].astype(np.float32)
    V_K = np.load(V_NPZ)["V_K"][:, :K].astype(np.float32)  # (512, 50)
    U_K = np.load(U_NPZ)["U_K"][:, :K].astype(np.float32)

    # ===== Per-(class, channel, patch) 512-D Haufe pattern =====
    # P_tok_full[c, ch, p, :] = Sigma_tok @ V_K @ W_tok[c, ch, p, :]
    #                         = V_K @ (eig_V * W_tok[c, ch, p, :])
    eig_W_tok = W_tok * eig_V                          # (4, 64, 4, 50) elementwise
    P_tok_full = np.einsum("dk,cnpk->cnpd", V_K, eig_W_tok)   # (4, 64, 4, 512)

    # ===== Per-class summary 512-D (sum over channel and patch) =====
    P_class_token_512 = P_tok_full.sum(axis=(1, 2))    # (4, 512) — token contribution only

    # ===== CLS contribution =====
    eig_W_cls = W_cls * eig_U                          # (4, 50)
    P_cls_class_512 = np.einsum("dk,ck->cd", U_K, eig_W_cls)   # (4, 512)

    # ===== Total per-class 512-D pattern =====
    P_total_class_512 = P_class_token_512 + P_cls_class_512    # (4, 512)

    # Per-(class, channel) summary 512-D (sum over patches)
    P_chan_class_512 = P_tok_full.sum(axis=2)          # (4, 64, 512)
    # Per-(class, patch) summary 512-D (sum over channels)
    P_patch_class_512 = P_tok_full.sum(axis=1)         # (4, 4, 512)

    print(f"P_total_class_512 shape: {P_total_class_512.shape}")
    print(f"P_class_token_512 shape: {P_class_token_512.shape}")
    print(f"P_cls_class_512   shape: {P_cls_class_512.shape}")

    np.savez_compressed(
        OUT_NPZ,
        W=W,
        eig_V=eig_V, eig_U=eig_U,
        P_tok_full=P_tok_full,                          # (4, 64, 4, 512)
        P_class_token_512=P_class_token_512,            # (4, 512)
        P_cls_class_512=P_cls_class_512,                # (4, 512)
        P_total_class_512=P_total_class_512,            # (4, 512)
        P_chan_class_512=P_chan_class_512,              # (4, 64, 512)
        P_patch_class_512=P_patch_class_512,            # (4, 4, 512)
        best_val=np.float32(best_val), test_bal=np.float32(test_bal),
    )
    print(f"saved {OUT_NPZ}", flush=True)

    # Identify top-loading 512-D coords per class
    print("\nTop-10 most-activated 512-D coords per class (|P_total|):")
    for c, name in enumerate(CLASS_NAMES):
        order = np.argsort(np.abs(P_total_class_512[c]))[::-1][:10]
        print(f"  {name}: {sorted(order.tolist())}    "
              f"max |P|={np.abs(P_total_class_512[c, order[0]]):.3g}")

    with PdfPages(OUT_PDF) as pdf:
        # ====== page 1: per-class total 512-D pattern (4 columns of P) ======
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        for c, name in enumerate(CLASS_NAMES):
            axes[c].bar(np.arange(D_EMB), P_total_class_512[c],
                        color=CLASS_COLORS[c], width=1.0)
            axes[c].set_ylabel(f"class {name}")
            axes[c].grid(True, alpha=0.3)
            top = np.argsort(np.abs(P_total_class_512[c]))[::-1][:8]
            for j in top:
                v = P_total_class_512[c, j]
                axes[c].text(j, v + 0.03 * np.sign(v) * np.abs(P_total_class_512[c]).max(),
                             str(j), ha="center", fontsize=7, rotation=90)
        axes[-1].set_xlabel("REVE embedding coord (0..511)")
        fig.suptitle(
            f"Haufe per-class 512-D pattern  (token + CLS contributions)\n"
            f"V/U probe test bal = {test_bal:.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ====== page 2: token vs CLS decomposition (overlay per class) ======
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        for c, name in enumerate(CLASS_NAMES):
            ax = axes[c]
            ax.plot(np.arange(D_EMB), P_class_token_512[c], color="C0",
                    linewidth=0.8, label="token")
            ax.plot(np.arange(D_EMB), P_cls_class_512[c], color="C3",
                    linewidth=0.8, label="CLS")
            ax.set_ylabel(f"class {name}")
            ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("REVE embedding coord (0..511)")
        fig.suptitle("Per-class 512-D pattern: token vs CLS contributions", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ====== page 3: per-(class, channel) heatmap (4 subplots, each 64 x 512) ======
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        vmax_chan = float(np.abs(P_chan_class_512).max())
        for c, name in enumerate(CLASS_NAMES):
            ax = axes[c // 2, c % 2]
            im = ax.imshow(P_chan_class_512[c], cmap="coolwarm",
                           vmin=-vmax_chan, vmax=vmax_chan,
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"class {name}")
            ax.set_xlabel("REVE coord (0..511)")
            ax.set_ylabel("EEG channel index (0..63)")
            plt.colorbar(im, ax=ax, fraction=0.025)
        fig.suptitle("Per-(class, channel) 512-D pattern (sum over patches)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ====== page 4: per-(class, patch) heatmap (4 subplots, each 4 x 512) ======
        fig, axes = plt.subplots(2, 2, figsize=(15, 6))
        vmax_p = float(np.abs(P_patch_class_512).max())
        for c, name in enumerate(CLASS_NAMES):
            ax = axes[c // 2, c % 2]
            im = ax.imshow(P_patch_class_512[c], cmap="coolwarm",
                           vmin=-vmax_p, vmax=vmax_p,
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"class {name}")
            ax.set_xlabel("REVE coord (0..511)")
            ax.set_ylabel("patch (0..3)")
            ax.set_yticks(range(N_PATCH))
            plt.colorbar(im, ax=ax, fraction=0.025)
        fig.suptitle("Per-(class, patch) 512-D pattern (sum over channels)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"wrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
