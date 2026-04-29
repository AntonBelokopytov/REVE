"""Extend the per-component spectrum analysis to ALL 512 V/U components.

Step 1: compute full V (512x512) from per-token uncentered second moment, full U (512x512) from per-window CLS uncentered second moment.
Step 2: stream over latents, project each token by V_full and each CLS by U_full ->
        per-window features (N, 256, 512) for tokens and (N, 512) for CLS.
Step 3: sweep k = 0..511, build single-component features (N, 257), train task and
        subject probes, record balanced accuracy.

Outputs:
  reve_VU_full.npz                 - V_full (512x512), U_full, eigvals
  reve_spectrum_full_512.npz       - task_bal[512], subject_bal[512]
  reve_spectrum_full_512.pdf       - 2-row plot: accuracy + eigvals (log)
  reve_spectrum_full_512.txt       - tabular results
"""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/media/alex/DATA1/REVE/hf_cache")
_TOK = Path("/media/alex/DATA1/REVE/.hf_token")
if _TOK.exists():
    os.environ.setdefault("HF_TOKEN", _TOK.read_text().strip())

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModel

LAT_DIR = Path("/media/alex/DATA1/REVE/latents_paper")
VU_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_VU_full.npz")
PROJ_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_proj_full.npz")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_spectrum_full_512.npz")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_spectrum_full_512.pdf")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_spectrum_full_512.txt")

D = 512
N_TOK = 256                     # 64 channels x 4 patches
EPOCHS = 20
BATCH = 64
LR = 1e-4
WD = 1e-2
DROPOUT = 0.1
PATIENCE = 5
SEED = 0


def softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------- step 1: compute V and U on all 512 dims ----------
def compute_VU():
    if VU_NPZ.exists():
        d = np.load(VU_NPZ)
        return d["V_full"], d["U_full"], d["eig_V"], d["eig_U"]

    print("Loading model for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(-1).astype(np.float64)
    scale = D ** -0.5
    del model

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    print(f"Streaming {len(files)} subjects to build M_token and M_cls ...", flush=True)
    M_tok = np.zeros((D, D), dtype=np.float64)
    M_cls = np.zeros((D, D), dtype=np.float64)
    n_tok = 0
    n_win = 0
    for i, p in enumerate(files, 1):
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)              # (N, C, P, D)
        N = lat.shape[0]
        flat = lat.reshape(N, N_TOK, D).astype(np.float64)
        # per-token
        toks = flat.reshape(-1, D)
        M_tok += toks.T @ toks
        n_tok += toks.shape[0]
        # CLS
        scores = (flat @ q) * scale
        w = softmax_np(scores, axis=1)
        cls = (w[..., None] * flat).sum(axis=1)                        # (N, D)
        M_cls += cls.T @ cls
        n_win += cls.shape[0]
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)} done", flush=True)

    M_tok = (M_tok + M_tok.T) * 0.5 / n_tok
    M_cls = (M_cls + M_cls.T) * 0.5 / n_win
    eig_V_, V_ = np.linalg.eigh(M_tok)
    eig_U_, U_ = np.linalg.eigh(M_cls)
    o_V = np.argsort(eig_V_)[::-1]
    o_U = np.argsort(eig_U_)[::-1]
    V_full = V_[:, o_V].astype(np.float32)                              # (D, D)
    U_full = U_[:, o_U].astype(np.float32)
    eig_V = eig_V_[o_V].astype(np.float32)
    eig_U = eig_U_[o_U].astype(np.float32)
    np.savez_compressed(VU_NPZ, V_full=V_full, U_full=U_full, eig_V=eig_V, eig_U=eig_U)
    print(f"  saved {VU_NPZ}", flush=True)
    return V_full, U_full, eig_V, eig_U


# ---------- step 2: project all tokens and CLSs into V/U coordinates ----------
def compute_projections(V_full, U_full):
    if PROJ_NPZ.exists():
        d = np.load(PROJ_NPZ)
        return d["proj_tokens"], d["proj_cls"], d["labels"], d["subjects"]

    print("Loading model for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(-1).astype(np.float32)
    scale = D ** -0.5
    del model

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    # First pass to know N_total
    n_total = 0
    for p in files:
        with h5py.File(p, "r") as f:
            n_total += f["trial/latent"].shape[0]
    print(f"Allocating proj_tokens fp16 ({n_total}, {N_TOK}, {D}) "
          f"= {n_total * N_TOK * D * 2 / 1e9:.2f} GB", flush=True)
    proj_tokens = np.empty((n_total, N_TOK, D), dtype=np.float16)
    proj_cls = np.empty((n_total, D), dtype=np.float32)
    labels = np.empty(n_total, dtype=np.int8)
    subjects = np.empty(n_total, dtype=np.int32)

    pos = 0
    V32 = V_full.astype(np.float32)
    U32 = U_full.astype(np.float32)
    for i, p in enumerate(files, 1):
        sid = int(Path(p).stem[1:])
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)
            lab = f["trial/label"][:]
        N = lat.shape[0]
        flat = lat.reshape(N, N_TOK, D)
        scores = (flat @ q) * scale
        w = softmax_np(scores, axis=1)
        cls = (w[..., None] * flat).sum(axis=1)                        # (N, D)
        proj_tokens[pos:pos + N] = (flat @ V32).astype(np.float16)
        proj_cls[pos:pos + N] = cls @ U32
        labels[pos:pos + N] = lab
        subjects[pos:pos + N] = sid
        pos += N
        if i % 20 == 0 or i == len(files):
            print(f"  proj {i}/{len(files)}", flush=True)

    np.savez(PROJ_NPZ,
             proj_tokens=proj_tokens, proj_cls=proj_cls,
             labels=labels, subjects=subjects)
    print(f"  saved {PROJ_NPZ}", flush=True)
    return proj_tokens, proj_cls, labels, subjects


# ---------- step 3: per-component probes ----------
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
            logits = model(xb); loss = F.cross_entropy(logits, yb)
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
    V_full, U_full, eig_V, eig_U = compute_VU()
    proj_tokens, proj_cls, labels, subjects = compute_projections(V_full, U_full)
    n = labels.shape[0]
    labels = labels.astype(np.int64)

    train_sids = np.arange(1, 71); val_sids = np.arange(71, 90); test_sids = np.arange(90, 110)
    m_tr = np.isin(subjects, train_sids); m_va = np.isin(subjects, val_sids); m_te = np.isin(subjects, test_sids)
    rng_split = np.random.default_rng(SEED)
    sub_split = np.empty(n, dtype=np.int8)
    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        rng_split.shuffle(idx)
        nn_ = len(idx); n_tr, n_va = int(0.7 * nn_), int(0.15 * nn_)
        sub_split[idx[:n_tr]] = 0
        sub_split[idx[n_tr:n_tr + n_va]] = 1
        sub_split[idx[n_tr + n_va:]] = 2
    s_tr = sub_split == 0; s_va = sub_split == 1; s_te = sub_split == 2
    uniq = np.unique(subjects)
    sid_to_class = {sid: i for i, sid in enumerate(uniq)}
    subject_y = np.array([sid_to_class[s] for s in subjects], dtype=np.int64)
    n_subjects = len(uniq)

    print(f"\ntask split    : tr={m_tr.sum()} va={m_va.sum()} te={m_te.sum()}", flush=True)
    print(f"subject split : tr={s_tr.sum()} va={s_va.sum()} te={s_te.sum()}  "
          f"({n_subjects}-way)", flush=True)
    print(f"sweeping k = 0..{D - 1} ...", flush=True)

    task_bal = np.zeros(D, dtype=np.float32)
    sub_bal = np.zeros(D, dtype=np.float32)
    t0 = time.time()
    for k in range(D):
        # slice: (N, 256) tokens at component k + (N,) cls at component k -> (N, 257)
        feat_k = np.empty((n, N_TOK + 1), dtype=np.float16)
        feat_k[:, 0] = proj_cls[:, k].astype(np.float16)
        feat_k[:, 1:] = proj_tokens[:, :, k]
        # task
        task_bal[k] = train_probe(
            feat_k[m_tr], labels[m_tr], feat_k[m_va], labels[m_va],
            feat_k[m_te], labels[m_te], 4,
        )
        # subject
        sub_bal[k] = train_probe(
            feat_k[s_tr], subject_y[s_tr], feat_k[s_va], subject_y[s_va],
            feat_k[s_te], subject_y[s_te], n_subjects,
        )
        if k < 5 or k == D - 1 or k % 25 == 0:
            print(f"  k={k:>3}  task={task_bal[k]:.3f}  subject={sub_bal[k]:.3f}  "
                  f"eig_V={eig_V[k]:.3g}  eig_U={eig_U[k]:.3g}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(OUT_NPZ, k=np.arange(D), task_bal=task_bal, subject_bal=sub_bal,
                        eig_V=eig_V, eig_U=eig_U)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(np.arange(D), task_bal, "-", color="C0", label="task (4-class)", linewidth=1)
    axes[0].plot(np.arange(D), sub_bal, "-", color="C3", label="subject (108-class)", linewidth=1)
    axes[0].axhline(0.25, color="C0", linestyle=":", alpha=0.4, label="task chance")
    axes[0].axhline(1 / n_subjects, color="C3", linestyle=":", alpha=0.4, label="subject chance")
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_title(f"Single-component spectrum: task vs subject across all {D} V/U components")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(np.arange(D), np.maximum(eig_V, 1e-30), color="C0", label="V eigvals (per-token)")
    axes[1].semilogy(np.arange(D), np.maximum(eig_U, 1e-30), color="C3", label="U eigvals (CLS)")
    axes[1].set_xlabel("component index k")
    axes[1].set_ylabel("eigenvalue (log)")
    axes[1].grid(True, which="both", alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")

    lines = ["k     task_bal  sub_bal   eig_V         eig_U"]
    for k in range(D):
        lines.append(f"{k:>4}    {task_bal[k]:.3f}    {sub_bal[k]:.3f}    "
                     f"{eig_V[k]:>10.3g}    {eig_U[k]:>10.3g}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_PDF}  {OUT_NPZ}  {OUT_TXT}")


if __name__ == "__main__":
    main()
