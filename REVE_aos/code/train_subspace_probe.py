"""Linear probe on REVE features projected onto the precomputed 50-D main subspaces.

For each window:
  per-token tokens  (256, 512)  -->  V_K coords (256, 50)        # V_K from main_subspace.npz
  attention-pooled CLS (512,)   -->  U_K coords (50,)            # U_K from cls_subspace.npz

Per-window feature: concat([cls_50, tokens_256x50.flatten()]) = 12,850-D vector
Head: RMSNorm + Dropout(0.1) + Linear(12850, 4)
Training: AdamW lr=1e-4 wd=1e-2, batch 32, 20 epochs, mixup off, patience 5
Split:  paper's fixed 70/19/20 (subjects 1..70 / 71..89 / 90..109)
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModel

LAT_DIR = Path("/media/alex/DATA1/REVE/latents_paper")
V_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_main_subspace.npz")
U_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_cls_subspace.npz")
FEAT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_subspace_features.npz")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_subspace_probe.txt")

D = 512
C = 64
P = 4
K = 50
N_TOK = C * P
N_CLS = 4
EPOCHS = 20
BATCH = 32
LR = 1e-4
WD = 1e-2
DROPOUT = 0.1
PATIENCE = 5
SEED = 0


def softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class SubspaceHead(nn.Module):
    """Operates on (B, F) where F = K + N_TOK * K = 50 + 12800 = 12850."""

    def __init__(self, n_features, n_classes=N_CLS, dropout=DROPOUT):
        super().__init__()
        self.norm = RMSNorm(n_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.fc(self.drop(self.norm(x)))


def build_features():
    """Build per-window features (N_total, K + N_TOK*K). Cached as fp16 in FEAT_NPZ."""
    if FEAT_NPZ.exists():
        d = np.load(FEAT_NPZ)
        return (d["features"], d["labels"], d["subjects"])

    print("Loading V_K and U_K ...", flush=True)
    V_K = np.load(V_NPZ)["V_K"][:, :K].astype(np.float32)        # (512, K)
    U_K = np.load(U_NPZ)["U_K"][:, :K].astype(np.float32)        # (512, K)
    print(f"  V_K shape {V_K.shape}, U_K shape {U_K.shape}")

    print("Loading model for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(-1).astype(np.float32)
    scale = D ** -0.5
    del model

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    feats, lbls, subs = [], [], []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        sid = int(Path(p).stem[1:])
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)         # (N, C, P, D)
            lab = f["trial/label"][:]
        N = lat.shape[0]

        # CLS via attention pool with HF cls_query_token
        flat = lat.reshape(N, N_TOK, D)                            # (N, 256, 512)
        scores = (flat @ q) * scale                                # (N, 256)
        w = softmax_np(scores, axis=1)
        cls = (w[..., None] * flat).sum(axis=1)                    # (N, 512)

        # Project tokens onto V_K: (N, 256, 512) @ (512, K) -> (N, 256, K)
        tok_proj = flat @ V_K                                       # (N, 256, K)
        # Project CLS onto U_K: (N, 512) @ (512, K) -> (N, K)
        cls_proj = cls @ U_K                                        # (N, K)

        feat = np.concatenate(
            [cls_proj, tok_proj.reshape(N, -1)], axis=1
        ).astype(np.float16)                                        # (N, K + N_TOK*K)

        feats.append(feat); lbls.append(lab)
        subs.append(np.full(N, sid, dtype=np.int32))
        if i % 20 == 0 or i == len(files):
            print(f"  built features {i}/{len(files)}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    features = np.concatenate(feats, axis=0)
    labels = np.concatenate(lbls)
    subjects = np.concatenate(subs)
    np.savez_compressed(FEAT_NPZ,
                        features=features, labels=labels, subjects=subjects,
                        K=np.int64(K))
    print(f"  saved features shape {features.shape} ({features.dtype}) -> {FEAT_NPZ}",
          flush=True)
    return features, labels, subjects


def stack(features, labels, subjects, sids):
    m = np.isin(subjects, list(sids))
    return features[m], labels[m].astype(np.int64), subjects[m]


def iterate(X, y, batch=BATCH, shuffle=True, rng=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        sl = idx[i:i + batch]
        yield torch.from_numpy(X[sl].astype(np.float32)), torch.from_numpy(y[sl])


def evaluate(model, X, y, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in iterate(X, y, shuffle=False):
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(-1).cpu().numpy())
            gts.append(yb.numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    return balanced_accuracy_score(gts, preds), (preds == gts).mean()


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    features, labels, subjects = build_features()
    print(f"\nfeatures: {features.shape} ({features.dtype})", flush=True)
    n_features = features.shape[1]

    train_sids = list(range(1, 71))
    val_sids   = list(range(71, 90))
    test_sids  = list(range(90, 110))
    Xtr, ytr, _ = stack(features, labels, subjects, train_sids)
    Xva, yva, _ = stack(features, labels, subjects, val_sids)
    Xte, yte, _ = stack(features, labels, subjects, test_sids)
    print(f"train/val/test = {len(ytr)}/{len(yva)}/{len(yte)} trials", flush=True)

    device = "cpu"
    model = SubspaceHead(n_features).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"head params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val = -1.0
    best_state = None
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0; n_batches = 0
        for xb, yb in iterate(Xtr, ytr, shuffle=True, rng=rng):
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item(); n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)

        val_bal, val_raw = evaluate(model, Xva, yva, device)
        elapsed = time.time() - t0
        improved = val_bal > best_val
        if improved:
            best_val = val_bal
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"epoch {epoch:>2}: train_loss={train_loss:.4f} val_bal={val_bal:.4f} "
              f"raw={val_raw:.4f} best_val={best_val:.4f}  "
              f"{'*' if improved else ' '}  elapsed={elapsed:.0f}s", flush=True)
        if bad_epochs >= PATIENCE:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_bal, test_raw = evaluate(model, Xte, yte, device)
    msg = (
        f"\nLinear probe on K=50 main-subspace projections of REVE latents:\n"
        f"  per-window features = U_K(CLS) || V_K(tokens) = {n_features}-D\n"
        f"  head params         = {n_params:,}\n"
        f"  train/val/test      = {len(ytr)}/{len(yva)}/{len(yte)} trials\n"
        f"  best val bal acc    = {best_val:.4f}\n"
        f"  test raw acc        = {test_raw:.4f}\n"
        f"  test balanced acc   = {test_bal:.4f}\n"
        f"\n  reference: full 131,584-D paper head -> 0.5794 (our run), 0.510 (paper)\n"
    )
    print(msg)
    OUT_TXT.write_text(msg)


if __name__ == "__main__":
    main()
