"""Haufe activation patterns of the K=50 V/U-subspace linear probe — efficient version.

The features the classifier sees live in V_K (per-token) and U_K (CLS) eigenbases.
Treating each (window, channel, patch) token as one observation (and each window as
one CLS observation), the uncentered covariance in those bases is diagonal with
values equal to the eigenvalues. So Haufe's formula

    P  =  Σ_g(f)  · Wᵀ        becomes        P[i, c] = σ_i · W[c, i]

elementwise, no 12850×12850 covariance to materialize. We then lift the per-slot
50-D pattern back into REVE's 512-D embedding space via V_K and U_K to obtain
the full 512-D activation patterns.

Steps:
  1. Train the V/U-subspace probe (paper split, RMSNorm + Linear).
  2. Take W (4, 12850) and build sigma (12850,):
        sigma[0:50]                   = eig_U[:50]              (CLS slots)
        sigma[50 + s·50 : 50 + (s+1)·50] = eig_V[:50] for s = 0..255  (token slots)
  3. Elementwise pattern P = sigma[:, None] · W^T   (12850, 4).
  4. Decompose:
        P_cls (50, 4)
        P_tok (64, 4, 50, 4) = (channel, patch, V-component, class)
  5. Lift to 512-D: per-(channel, class) 512-D pattern = V_K @ Σ_patch P_tok[ch, p, :, c].
  6. Visualize:
        a) per-class topomap (sum over patches + V-components, signed)
        b) per-class topomap for each top-task V-component (k = 29, 37, 41)
        c) CLS pattern bars per class
        d) per-V-component pattern energy per class
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import balanced_accuracy_score

mne.set_log_level("ERROR")

FEAT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_subspace_features.npz")
V_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_main_subspace.npz")
U_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_cls_subspace.npz")
LAT_DIR = Path("/media/alex/DATA1/REVE/latents_paper")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_haufe_W.npz")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_haufe.pdf")

D_EMB = 512
K = 50
N_CHAN = 64
N_PATCH = 4
N_TOK = N_CHAN * N_PATCH                  # 256
N_FEAT = K + N_TOK * K                    # 12850
N_CLS = 4
EPOCHS = 20
BATCH = 32
LR = 1e-4
WD = 1e-2
DROPOUT = 0.1
PATIENCE = 5
SEED = 0
CLASS_NAMES = ["L", "R", "B", "F"]
TOP_TASK_K = [29, 37, 41]


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
        print(f"  epoch {epoch:>2}: train_loss={loss_sum/nb:.4f}  val_bal={val_bal:.4f}  "
              f"best={best_val:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_val


def main():
    d = np.load(FEAT_NPZ)
    features = d["features"]
    labels = d["labels"].astype(np.int64)
    subjects = d["subjects"]

    train_sids = list(range(1, 71))
    val_sids = list(range(71, 90))
    test_sids = list(range(90, 110))
    m_tr = np.isin(subjects, train_sids)
    m_va = np.isin(subjects, val_sids)
    m_te = np.isin(subjects, test_sids)
    print(f"train/val/test = {m_tr.sum()}/{m_va.sum()}/{m_te.sum()}", flush=True)

    print("Training V/U probe ...", flush=True)
    model, best_val = train_probe(features[m_tr], labels[m_tr], features[m_va], labels[m_va])
    test_bal, test_raw = evaluate(model, features[m_te], labels[m_te])
    print(f"best val = {best_val:.4f}   test bal = {test_bal:.4f}   raw = {test_raw:.4f}",
          flush=True)

    W = model.fc.weight.detach().numpy()      # (4, 12850)
    b = model.fc.bias.detach().numpy()
    gamma = model.norm.weight.detach().numpy()

    eig_V = np.load(V_NPZ)["eigvals"][:K].astype(np.float32)
    eig_U = np.load(U_NPZ)["eigvals"][:K].astype(np.float32)
    V_K = np.load(V_NPZ)["V_K"][:, :K].astype(np.float32)             # (512, 50)
    U_K = np.load(U_NPZ)["U_K"][:, :K].astype(np.float32)             # (512, 50)

    # Build sigma diagonal in feature space
    sigma = np.empty(N_FEAT, dtype=np.float32)
    sigma[:K] = eig_U
    for s in range(N_TOK):
        sigma[K + s * K:K + (s + 1) * K] = eig_V

    # Haufe pattern: P[i, c] = sigma_i * W[c, i]
    P = sigma[:, None] * W.T                                           # (12850, 4)
    P_cls = P[:K, :]                                                    # (50, 4)
    P_tok = P[K:, :].reshape(N_CHAN, N_PATCH, K, N_CLS)                 # (64, 4, 50, 4)

    # ---------- 512-D lifted patterns per (channel, class) ----------
    # For each (channel, class): sum over patches of P_tok -> (50,)
    # then lift via V_K  ->  (512,)
    P_chan_50 = P_tok.sum(axis=1)                                       # (64, 50, 4)
    P_chan_512 = np.einsum("dk,ckl->cdl", V_K, P_chan_50)                # (64, 512, 4)

    # Per class summary 512-D (sum across channels too)
    P_class_512 = P_chan_512.sum(axis=0)                                # (512, 4)

    # Channel info
    files = sorted([p for p in LAT_DIR.glob("S*.h5")])
    with h5py.File(files[0], "r") as f:
        ch_names = [c.decode() for c in f.attrs["channel_names"]]
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")

    np.savez_compressed(
        OUT_NPZ,
        W=W, b=b, gamma=gamma, P=P, P_cls=P_cls,
        P_tok=P_tok.reshape(N_CHAN * N_PATCH * K, N_CLS),
        P_chan_512=P_chan_512.reshape(N_CHAN * D_EMB, N_CLS),
        P_class_512=P_class_512,
        ch_names=np.array(ch_names, dtype="S16"),
        eig_V=eig_V, eig_U=eig_U,
        best_val=np.float32(best_val), test_bal=np.float32(test_bal),
    )
    print(f"saved {OUT_NPZ}", flush=True)

    # Per-class topomap signal: sum over patches & V-components -> (channels, classes)
    topo_per_class = P_tok.sum(axis=(1, 2))                             # (64, 4)

    with PdfPages(OUT_PDF) as pdf:
        # Page 1: per-class topomap
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
        vmax = float(np.abs(topo_per_class).max())
        for c, name in enumerate(CLASS_NAMES):
            mne.viz.plot_topomap(
                topo_per_class[:, c], info, axes=axes[c], show=False,
                cmap="coolwarm", vlim=(-vmax, vmax),
                contours=4, sensors=True,
            )
            axes[c].set_title(name)
        fig.suptitle(
            f"Haufe per-class topomap  (sum over 4 patches × 50 V-components, clip ±{vmax:.2g})\n"
            f"V/U probe test bal = {test_bal:.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: per top-task V-component, per-class topomap
        fig, axes = plt.subplots(len(TOP_TASK_K), 4,
                                 figsize=(15, 4.0 * len(TOP_TASK_K)))
        for r, kk in enumerate(TOP_TASK_K):
            row = P_tok[:, :, kk, :].sum(axis=1)                        # (64, 4)
            vmax_r = float(np.abs(row).max())
            for c, name in enumerate(CLASS_NAMES):
                ax = axes[r, c]
                mne.viz.plot_topomap(
                    row[:, c], info, axes=ax, show=False, cmap="coolwarm",
                    vlim=(-vmax_r, vmax_r), contours=4, sensors=True,
                )
                if r == 0:
                    ax.set_title(name)
                if c == 0:
                    ax.set_ylabel(f"V-comp k={kk}\n(eig={eig_V[kk]:.2g})", fontsize=10)
        fig.suptitle("Haufe topo per top-task V-component  (sum over 4 patches)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3: CLS pattern bars + V-component pattern energy
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
        ax = axes[0]
        x = np.arange(K); width = 0.2
        for c, name in enumerate(CLASS_NAMES):
            ax.bar(x + (c - 1.5) * width, P_cls[:, c], width=width, label=name)
        ax.set_xlabel("U-component k")
        ax.set_ylabel("CLS pattern weight")
        ax.set_title("CLS pattern P_cls per class (top 50 U-components)")
        ax.legend(loc="best"); ax.grid(True, alpha=0.3)

        ax = axes[1]
        comp_energy = np.linalg.norm(P_tok, axis=(0, 1))                # (50, 4)
        for c, name in enumerate(CLASS_NAMES):
            ax.plot(np.arange(K), comp_energy[:, c], "-o", markersize=3, label=name)
        for kk in TOP_TASK_K:
            ax.axvline(kk, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlabel("V-component k")
        ax.set_ylabel("||token pattern||₂ per V-component")
        ax.set_title("Token pattern energy per V-component  (vertical lines: top-task k)")
        ax.legend(loc="best"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 4: per-class 512-D pattern (in REVE embedding space, summed over channels)
        fig, ax = plt.subplots(figsize=(13, 4.5))
        for c, name in enumerate(CLASS_NAMES):
            ax.plot(np.arange(D_EMB), P_class_512[:, c], "-", linewidth=0.8, label=name)
        ax.set_xlabel("REVE embedding coord (0..511)")
        ax.set_ylabel("class-summary 512-D pattern (sum over channels)")
        ax.set_title("Per-class 512-D pattern in REVE embedding space  "
                     "(V_K-lifted, summed over 64 channels and 4 patches)")
        ax.legend(loc="best"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"wrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
