"""Pool preprocessed REVE tokens, then run LOSO linear probe.

Reads preprocessed-pipeline tokens from /media/alex/DATA1/REVE/latents_pp/.
Applies REVE's attention_pooling head (cls_query_token) for the per-window vector.
Then LOSO 4-class linear probe on PCA-50 features.

Targets paper Table 4: REVE-B (Pool) on PhysioNetMI = 0.537 +- 0.005 balanced acc.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

LAT_DIR = Path("/media/alex/DATA1/REVE/latents_pp")
EMB_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_pertrial_embeddings_pp.npz")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_linear_probe_pp.txt")
PCA_K = 50
LR_C = 1.0
D = 512


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def pool_all():
    print("Loading REVE for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(-1)
    scale = D ** -0.5
    del model

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    embs, lbls, subs = [], [], []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        sid = int(Path(p).stem[1:])
        with h5py.File(p, "r") as f:
            t = f["windows/trial/latent"][:].astype(np.float32)
            l = f["windows/trial/label"][:]
        N, C, P, _ = t.shape
        flat = t.reshape(N, C * P, D)
        scores = (flat @ q) * scale
        w = softmax(scores, axis=1)
        emb = (w[..., None] * flat).sum(axis=1).astype(np.float32)
        embs.append(emb); lbls.append(l); subs.append(np.full(N, sid, dtype=np.int32))
        if i % 20 == 0 or i == len(files):
            print(f"  pooled {i}/{len(files)}  elapsed={time.time()-t0:.0f}s", flush=True)

    embs = np.concatenate(embs)
    lbls = np.concatenate(lbls)
    subs = np.concatenate(subs)
    np.savez_compressed(EMB_NPZ, embeddings=embs, labels=lbls, subjects=subs)
    print(f"  saved {EMB_NPZ}  shape={embs.shape}", flush=True)
    return embs, lbls, subs


def loso_probe(emb, lbl, sub):
    m = lbl >= 0
    X, y, s = emb[m], lbl[m].astype(int), sub[m]
    print(f"\n{len(y)} motor trials, {len(np.unique(s))} subjects", flush=True)

    subjects = np.unique(s)
    accs = np.zeros(len(subjects))
    bal_accs = np.zeros(len(subjects))
    t0 = time.time()
    for i, sid in enumerate(subjects):
        tr = s != sid
        te = s == sid
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        pca = PCA(n_components=PCA_K, svd_solver="randomized", random_state=0).fit(Xtr)
        Ztr = pca.transform(Xtr); Zte = pca.transform(Xte)
        clf = LogisticRegression(C=LR_C, max_iter=500, solver="lbfgs", tol=1e-4)
        clf.fit(Ztr, y[tr])
        accs[i] = clf.score(Zte, y[te])
        bal_accs[i] = balanced_accuracy_score(y[te], clf.predict(Zte))
        if i < 3 or (i + 1) % 20 == 0 or i == len(subjects) - 1:
            print(f"  fold {i+1:3d}/{len(subjects)}  S{sid:03d}  "
                  f"acc={accs[i]:.3f}  bal_acc={bal_accs[i]:.3f}  "
                  f"running mean acc={accs[:i+1].mean():.3f}  "
                  f"bal={bal_accs[:i+1].mean():.3f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    msg = (
        f"\nLOSO 4-class linear probe on PREPROCESSED REVE per-window embeddings:\n"
        f"  N folds (subjects) = {len(accs)}\n"
        f"  features = top-{PCA_K} PCA components of attention-pooled 512-D\n"
        f"  raw acc       = {accs.mean():.3f}  +/-  {accs.std():.3f}\n"
        f"  balanced acc  = {bal_accs.mean():.3f}  +/-  {bal_accs.std():.3f}\n"
        f"  median bal    = {np.median(bal_accs):.3f}\n"
        f"  min/max bal   = {bal_accs.min():.3f} / {bal_accs.max():.3f}\n"
        f"\n  paper Table 4 REVE-B (Pool) on PhysioNetMI:  0.537 +/- 0.005 (balanced acc)\n"
        f"  paper Table 4 REVE-B          on PhysioNetMI:  0.510 +/- 0.012\n"
    )
    print(msg)
    OUT_TXT.write_text(msg)
    print(f"wrote {OUT_TXT}", flush=True)


def main():
    if EMB_NPZ.exists():
        print(f"loading cached pooled embeddings from {EMB_NPZ}")
        d = np.load(EMB_NPZ)
        emb, lbl, sub = d["embeddings"], d["labels"], d["subjects"]
    else:
        emb, lbl, sub = pool_all()
    loso_probe(emb, lbl, sub)


if __name__ == "__main__":
    main()
