"""Train REVE's paper "linear probe" head on PhysioNet-MI.

Reproduces the protocol from reve_eeg/src/configs/task/physio.yaml +
reve_eeg/src/models/classifier.py with pooling="no":
  - frozen encoder (we use pre-saved latents from extract_latents_paper.py)
  - learnable cls_query_token (B,1,D) for attention pooling
  - context = softmax(q . tokens / sqrt(D)) @ tokens  -> (B,1,D)
  - cat([context, tokens], dim=-2) -> (B, 1+C*P, D), flatten -> (B, (1+C*P)*D)
  - RMSNorm + Dropout(0.1) + Linear((1+C*P)*D, n_classes)
  - AdamW lr=1e-4, wd=1e-2, batch 32, 20 epochs, mixup off, patience 5
  - fixed split: subjects 1..70 train, 71..89 val, 90..109 test
"""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

LAT_DIR = Path("/media/alex/DATA1/REVE/latents_paper")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_paper_probe.txt")

D = 512                         # embed_dim
C = 64                          # channels
P = 4                           # patches per window
N_TOK = C * P                   # 256
N_CLS = 4
EPOCHS = 20
BATCH = 32
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


class PaperHead(nn.Module):
    """pooling='no' head from classifier.py."""

    def __init__(self, embed_dim=D, n_tokens=N_TOK, n_classes=N_CLS, dropout=DROPOUT):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        out_shape = (n_tokens + 1) * embed_dim
        self.norm = RMSNorm(out_shape)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_shape, n_classes)

    def forward(self, x):                                  # x: (B, N, D)
        b = x.shape[0]
        q = self.cls_query_token.expand(b, -1, -1)         # (B,1,D)
        scores = (q @ x.transpose(-1, -2)) / (self.embed_dim ** 0.5)  # (B,1,N)
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ x                                      # (B,1,D)
        z = torch.cat([ctx, x], dim=-2)                     # (B,N+1,D)
        z = z.reshape(b, -1)                                # (B,(N+1)*D)
        z = self.norm(z)
        z = self.drop(z)
        return self.fc(z)


def load_subjects():
    """Returns dict sid -> (latents (N,N_TOK,D) fp16, labels (N,) int).

    Keep fp16 in memory to halve footprint (~2.3 GB instead of 4.5 GB).
    Per-batch upcast to fp32 happens in iterate().
    """
    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    out = {}
    for p in files:
        sid = int(Path(p).stem[1:])
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:]                            # native fp16
            lbl = f["trial/label"][:].astype(np.int64)
        N = lat.shape[0]
        out[sid] = (lat.reshape(N, N_TOK, D), lbl)
    return out


def stack(subjects, sids):
    Xs, ys = [], []
    for sid in sids:
        if sid not in subjects:
            continue
        x, y = subjects[sid]
        Xs.append(x); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def iterate(X, y, batch=BATCH, shuffle=True, rng=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        sl = idx[i:i + batch]
        # upcast fp16 -> fp32 per batch
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

    print("Loading per-subject latents ...", flush=True)
    subjects = load_subjects()
    print(f"  {len(subjects)} subjects loaded", flush=True)

    train_sids = list(range(1, 71))
    val_sids   = list(range(71, 90))
    test_sids  = list(range(90, 110))
    Xtr, ytr = stack(subjects, train_sids)
    Xva, yva = stack(subjects, val_sids)
    Xte, yte = stack(subjects, test_sids)
    print(f"train: N={len(ytr)} classes={dict(zip(*np.unique(ytr, return_counts=True)))}")
    print(f"val:   N={len(yva)} classes={dict(zip(*np.unique(yva, return_counts=True)))}")
    print(f"test:  N={len(yte)} classes={dict(zip(*np.unique(yte, return_counts=True)))}")

    device = "cpu"
    model = PaperHead().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"head params: {n_params:,}")

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
            loss_sum += float(loss); n_batches += 1
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
        print(f"epoch {epoch:>2}: train_loss={train_loss:.4f}  val_bal={val_bal:.4f} "
              f"raw={val_raw:.4f}  best_val={best_val:.4f}  "
              f"{'*' if improved else ' '}  elapsed={elapsed:.0f}s", flush=True)
        if bad_epochs >= PATIENCE:
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_bal, test_raw = evaluate(model, Xte, yte, device)
    msg = (
        f"\nPaper-protocol linear probe on PhysioNet-MI:\n"
        f"  pre-pool head: pooling='no'  (1 + 256 tokens flattened, RMSNorm + Linear)\n"
        f"  train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}  trials\n"
        f"  best val bal acc: {best_val:.4f}\n"
        f"  test raw acc      : {test_raw:.4f}\n"
        f"  test balanced acc : {test_bal:.4f}\n"
        f"\n  paper Table 4 REVE-B: 0.510 +- 0.012 (balanced)\n"
        f"  paper Table 4 REVE-B (Pool): 0.537 +- 0.005 (balanced)\n"
    )
    print(msg)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(msg)


if __name__ == "__main__":
    main()
