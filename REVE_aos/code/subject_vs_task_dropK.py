"""Probe subject identity vs. motor-imagery class as a function of "drop top-K' components".

Take the cached 50-D V_K and 50-D U_K projections of REVE latents. For each K' in
{0, 1, 2, 3, 5, 10, 20, 30, 40, 49}:
  - Drop the first K' components from the per-token V projection AND the per-window
    U projection (CLS).
  - Train two probes on the residual features:
      task    : 4-class motor imagery,  subjects 1..70 train / 71..89 val / 90..109 test
      subject : 108-class subject ID,  80/20 within-subject random split
  - Report test accuracy of each probe.

Hypothesis: the dominant V/U components encode subject identity. If true:
  subject accuracy collapses fast with K'
  task    accuracy stays stable (or even improves)
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
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_subject_vs_task_dropK.npz")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_subject_vs_task_dropK.pdf")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_subject_vs_task_dropK.txt")

K_FULL = 50
N_TOK = 256
EPOCHS = 20
BATCH = 64
LR = 1e-4
WD = 1e-2
DROPOUT = 0.1
PATIENCE = 5
SEED = 0
K_DROP_GRID = [0, 1, 2, 3, 5, 10, 20, 30, 40, 49]


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


def make_residual_features(features, k_drop):
    """features: (N, K_FULL + N_TOK*K_FULL)  ->  drop first k_drop coords of CLS and each token."""
    n = features.shape[0]
    cls = features[:, :K_FULL]                               # (N, 50)
    tok = features[:, K_FULL:].reshape(n, N_TOK, K_FULL)     # (N, 256, 50)
    cls_kept = cls[:, k_drop:]                               # (N, 50-k_drop)
    tok_kept = tok[:, :, k_drop:]                            # (N, 256, 50-k_drop)
    out = np.concatenate([cls_kept, tok_kept.reshape(n, -1)], axis=1).astype(np.float16)
    return out


def iterate(X, y, batch=BATCH, shuffle=True, rng=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        sl = idx[i:i + batch]
        yield torch.from_numpy(X[sl].astype(np.float32)), torch.from_numpy(y[sl])


def train_probe(Xtr, ytr, Xva, yva, Xte, yte, n_classes, label):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cpu"
    n_features = Xtr.shape[1]
    model = Head(n_features, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in iterate(Xtr, ytr, shuffle=True, rng=rng):
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        # eval val
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in iterate(Xva, yva, shuffle=False):
                preds.append(model(xb.to(device)).argmax(-1).cpu().numpy())
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
        for xb, yb in iterate(Xte, yte, shuffle=False):
            preds.append(model(xb.to(device)).argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    te_bal = balanced_accuracy_score(yte, preds)
    te_raw = (preds == yte).mean()
    return float(best_val), float(te_bal), float(te_raw)


def main():
    d = np.load(FEAT_NPZ)
    features = d["features"].astype(np.float16)
    labels = d["labels"].astype(np.int64)
    subjects = d["subjects"]

    # ----- task split (paper) -----
    train_sids = list(range(1, 71)); val_sids = list(range(71, 90)); test_sids = list(range(90, 110))
    m_tr = np.isin(subjects, train_sids); m_va = np.isin(subjects, val_sids); m_te = np.isin(subjects, test_sids)

    # ----- subject split: stratified within-subject 70/15/15 -----
    rng_split = np.random.default_rng(SEED)
    sub_split = np.empty(len(subjects), dtype=np.int8)  # 0=train,1=val,2=test
    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        rng_split.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(0.7 * n), int(0.15 * n)
        sub_split[idx[:n_tr]] = 0
        sub_split[idx[n_tr:n_tr + n_va]] = 1
        sub_split[idx[n_tr + n_va:]] = 2
    s_tr = sub_split == 0; s_va = sub_split == 1; s_te = sub_split == 2
    # convert subject IDs to dense 0..107 indices
    uniq = np.unique(subjects)
    sid_to_class = {sid: i for i, sid in enumerate(uniq)}
    subject_y = np.array([sid_to_class[s] for s in subjects], dtype=np.int64)
    n_subjects = len(uniq)

    print(f"task split:    train={m_tr.sum()} val={m_va.sum()} test={m_te.sum()}", flush=True)
    print(f"subject split: train={s_tr.sum()} val={s_va.sum()} test={s_te.sum()}  ({n_subjects}-way)", flush=True)
    print()

    rows = []
    t0 = time.time()
    for k_drop in K_DROP_GRID:
        feats_k = make_residual_features(features, k_drop)
        n_features = feats_k.shape[1]

        # task
        Xtr, ytr = feats_k[m_tr], labels[m_tr]
        Xva, yva = feats_k[m_va], labels[m_va]
        Xte, yte = feats_k[m_te], labels[m_te]
        task_val, task_te_bal, task_te_raw = train_probe(
            Xtr, ytr, Xva, yva, Xte, yte, 4, f"task k_drop={k_drop}"
        )

        # subject
        Xtr_s, ytr_s = feats_k[s_tr], subject_y[s_tr]
        Xva_s, yva_s = feats_k[s_va], subject_y[s_va]
        Xte_s, yte_s = feats_k[s_te], subject_y[s_te]
        sub_val, sub_te_bal, sub_te_raw = train_probe(
            Xtr_s, ytr_s, Xva_s, yva_s, Xte_s, yte_s, n_subjects, f"subject k_drop={k_drop}"
        )

        rows.append((k_drop, n_features, task_te_bal, task_te_raw, sub_te_bal, sub_te_raw))
        print(f"  k_drop={k_drop:>2}  feat_dim={n_features:>6}  "
              f"task_bal={task_te_bal:.3f} (raw {task_te_raw:.3f})  "
              f"subject_bal={sub_te_bal:.3f} (raw {sub_te_raw:.3f})  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    rows = np.array(rows, dtype=object)
    np.savez_compressed(
        OUT_NPZ,
        K_drop=np.array([r[0] for r in rows]),
        feat_dim=np.array([r[1] for r in rows]),
        task_bal=np.array([r[2] for r in rows], dtype=np.float32),
        task_raw=np.array([r[3] for r in rows], dtype=np.float32),
        subject_bal=np.array([r[4] for r in rows], dtype=np.float32),
        subject_raw=np.array([r[5] for r in rows], dtype=np.float32),
    )

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    K_arr = np.array([r[0] for r in rows])
    task_bal = np.array([r[2] for r in rows])
    sub_bal = np.array([r[4] for r in rows])
    ax.plot(K_arr, sub_bal, "-o", label="subject (108-class) balanced acc", color="C3")
    ax.plot(K_arr, task_bal, "-o", label="task (4-class) balanced acc", color="C0")
    ax.axhline(1 / n_subjects, color="C3", linestyle=":", alpha=0.4, label=f"chance subject = {1/n_subjects:.4f}")
    ax.axhline(0.25, color="C0", linestyle=":", alpha=0.4, label="chance task = 0.25")
    ax.set_xlabel("K' = number of dropped top components")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Subject vs task decodability vs dropping top-K' components")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")

    lines = ["k_drop  feat_dim   task_bal  task_raw   sub_bal  sub_raw"]
    for r in rows:
        lines.append(f"{r[0]:>5}  {r[1]:>8}   {r[2]:>7.3f}   {r[3]:>6.3f}   {r[4]:>6.3f}   {r[5]:>6.3f}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\nwrote {OUT_PDF}  {OUT_NPZ}  {OUT_TXT}")


if __name__ == "__main__":
    main()
