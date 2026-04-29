"""Where in the V/U spectrum does task signal live? Subject signal?

For each k in 0..49:
  features = [cls_coord_k (1-D), tok_coord_k for each of 256 tokens (256-D)] = 257-D
  Train task probe (4-class, paper between-subject split) -> test bal acc.
  Train subject probe (108-class, within-subject 70/15/15) -> test bal acc.

Plot task_bal(k) and subject_bal(k) vs k.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

FEAT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_subspace_features.npz")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_single_component_spectrum.npz")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_single_component_spectrum.pdf")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_single_component_spectrum.txt")

K_FULL = 50
N_TOK = 256
EPOCHS = 20
BATCH = 64
LR = 1e-4
WD = 1e-2
DROPOUT = 0.1
PATIENCE = 5
SEED = 0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class Head(nn.Module):
    def __init__(self, n_features, n_classes, dropout=DROPOUT):
        super().__init__()
        self.norm = RMSNorm(n_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.fc(self.drop(self.norm(x)))


def single_component(features, k):
    """Keep only component k of CLS and tokens. Returns (N, 257)."""
    n = features.shape[0]
    cls_k = features[:, k:k + 1]                                       # (N, 1)
    tok = features[:, K_FULL:].reshape(n, N_TOK, K_FULL)
    tok_k = tok[:, :, k]                                                # (N, 256)
    return np.concatenate([cls_k, tok_k], axis=1).astype(np.float16)


def iterate(X, y, batch=BATCH, shuffle=True, rng=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        sl = idx[i:i + batch]
        yield torch.from_numpy(X[sl].astype(np.float32)), torch.from_numpy(y[sl])


def train_probe(Xtr, ytr, Xva, yva, Xte, yte, n_classes):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    n_features = Xtr.shape[1]
    model = Head(n_features, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state, bad = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in iterate(Xtr, ytr, shuffle=True, rng=rng):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in iterate(Xva, yva, shuffle=False):
                preds.append(model(xb).argmax(-1).numpy())
        preds = np.concatenate(preds)
        val_bal = balanced_accuracy_score(yva, preds)
        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in iterate(Xte, yte, shuffle=False):
            preds.append(model(xb).argmax(-1).numpy())
    preds = np.concatenate(preds)
    return float(balanced_accuracy_score(yte, preds))


def main():
    d = np.load(FEAT_NPZ)
    features = d["features"].astype(np.float16)
    labels = d["labels"].astype(np.int64)
    subjects = d["subjects"]

    train_sids = list(range(1, 71))
    val_sids = list(range(71, 90))
    test_sids = list(range(90, 110))
    m_tr = np.isin(subjects, train_sids)
    m_va = np.isin(subjects, val_sids)
    m_te = np.isin(subjects, test_sids)

    rng_split = np.random.default_rng(SEED)
    sub_split = np.empty(len(subjects), dtype=np.int8)
    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        rng_split.shuffle(idx)
        n = len(idx); n_tr, n_va = int(0.7 * n), int(0.15 * n)
        sub_split[idx[:n_tr]] = 0
        sub_split[idx[n_tr:n_tr + n_va]] = 1
        sub_split[idx[n_tr + n_va:]] = 2
    s_tr = sub_split == 0; s_va = sub_split == 1; s_te = sub_split == 2
    uniq = np.unique(subjects)
    sid_to_class = {sid: i for i, sid in enumerate(uniq)}
    subject_y = np.array([sid_to_class[s] for s in subjects], dtype=np.int64)
    n_subjects = len(uniq)

    # also load eigvals for context
    V = np.load(Path("/media/alex/DATA1/REVE/reports/reve_main_subspace.npz"))
    U = np.load(Path("/media/alex/DATA1/REVE/reports/reve_cls_subspace.npz"))
    eig_V = V["eigvals"][:K_FULL]
    eig_U = U["eigvals"][:K_FULL]

    print(f"task split: train={m_tr.sum()} val={m_va.sum()} test={m_te.sum()}", flush=True)
    print(f"subject split: train={s_tr.sum()} val={s_va.sum()} test={s_te.sum()}  ({n_subjects}-way)", flush=True)
    print(f"  V eigvals top-10: {eig_V[:10].round(2).tolist()}")
    print(f"  U eigvals top-10: {eig_U[:10].round(2).tolist()}")
    print()

    task_curve = np.zeros(K_FULL)
    sub_curve = np.zeros(K_FULL)
    t0 = time.time()
    for k in range(K_FULL):
        Xk = single_component(features, k)
        # task
        task_curve[k] = train_probe(
            Xk[m_tr], labels[m_tr], Xk[m_va], labels[m_va], Xk[m_te], labels[m_te], 4,
        )
        # subject
        sub_curve[k] = train_probe(
            Xk[s_tr], subject_y[s_tr], Xk[s_va], subject_y[s_va],
            Xk[s_te], subject_y[s_te], n_subjects,
        )
        elapsed = time.time() - t0
        print(f"  k={k:>2}  task={task_curve[k]:.3f}  subject={sub_curve[k]:.3f}  "
              f"elapsed={elapsed:.0f}s", flush=True)

    np.savez_compressed(
        OUT_NPZ,
        k=np.arange(K_FULL), task_bal=task_curve.astype(np.float32),
        subject_bal=sub_curve.astype(np.float32),
        eig_V=eig_V.astype(np.float32), eig_U=eig_U.astype(np.float32),
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    ax.plot(np.arange(K_FULL), task_curve, "-o", color="C0", label="task (4-class)")
    ax.plot(np.arange(K_FULL), sub_curve, "-o", color="C3", label="subject (108-class)")
    ax.axhline(0.25, color="C0", linestyle=":", alpha=0.5, label="task chance")
    ax.axhline(1 / n_subjects, color="C3", linestyle=":", alpha=0.5, label="subject chance")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Where in V/U spectrum does each signal live?  (single component k)")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(np.arange(K_FULL), eig_V, "-o", color="C0", label="V eigvals (per-token)")
    ax.semilogy(np.arange(K_FULL), eig_U, "-o", color="C3", label="U eigvals (CLS)")
    ax.set_xlabel("component index k")
    ax.set_ylabel("eigenvalue (log)")
    ax.legend(loc="best"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")

    lines = ["k   task_bal   sub_bal   eig_V         eig_U"]
    for k in range(K_FULL):
        lines.append(f"{k:>2}    {task_curve[k]:.3f}    {sub_curve[k]:.3f}    "
                     f"{eig_V[k]:>10.3g}    {eig_U[k]:>10.3g}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\nwrote {OUT_PDF}  {OUT_NPZ}  {OUT_TXT}")


if __name__ == "__main__":
    main()
