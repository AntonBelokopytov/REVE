"""Try different pooling strategies on saved REVE per-token tokens.

The paper's 0.510 isn't reproduced with attention_pooling -> 0.352. Test alternatives:
  attention   : softmax(q . token) @ tokens   (REVE's pretraining pooling, baseline)
  mean        : tokens.mean over (C, P)        -> 512-D
  mean_patch  : tokens.mean over P            -> 64*512 = 32768-D
  mean_chan   : tokens.mean over C            -> 4*512 = 2048-D
  flat        : flatten C*P*D                 -> 131072-D (skipped, too big)
For each pooled representation, run LOSO 4-class linear probe with PCA-K + LR.
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
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

LAT_DIR = Path("/media/alex/DATA1/REVE/latents")
OUT = Path("/media/alex/DATA1/REVE/reports/reve_linear_probe_pooling.txt")
PCA_K = 50
LR_C = 1.0


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def build_features():
    print("Loading model for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(-1)
    scale = q.shape[0] ** -0.5
    del model

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    pools = {"attention": [], "mean": [], "mean_patch": [], "mean_chan": []}
    labels, subjects = [], []

    for i, p in enumerate(files, 1):
        sid = int(Path(p).stem[1:])
        with h5py.File(p, "r") as f:
            t = f["windows/trial/latent"][:].astype(np.float32)   # (N, C, P, D)
            l = f["windows/trial/label"][:]
        N, C, P, D = t.shape
        flat = t.reshape(N, C * P, D)
        # attention
        scores = (flat @ q) * scale
        w = softmax(scores, axis=1)
        pools["attention"].append((w[..., None] * flat).sum(axis=1).astype(np.float32))
        # mean over all tokens
        pools["mean"].append(t.reshape(N, -1, D).mean(axis=1).astype(np.float32))
        # mean over P -> per-channel features (N, C*D)
        pools["mean_patch"].append(t.mean(axis=2).reshape(N, -1).astype(np.float32))
        # mean over C -> per-patch features (N, P*D)
        pools["mean_chan"].append(t.mean(axis=1).reshape(N, -1).astype(np.float32))
        labels.append(l); subjects.append(np.full(N, sid, dtype=np.int32))
        if i % 20 == 0 or i == len(files):
            print(f"  built features {i}/{len(files)}", flush=True)

    pools = {k: np.concatenate(v) for k, v in pools.items()}
    labels = np.concatenate(labels); subjects = np.concatenate(subjects)
    return pools, labels, subjects


def loso_probe(X, y, s, pca_k, name):
    subjects = np.unique(s)
    accs = np.zeros(len(subjects))
    t0 = time.time()
    for i, sid in enumerate(subjects):
        tr = s != sid; te = s == sid
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        K = min(pca_k, len(tr) - 1, X.shape[1])
        pca = PCA(n_components=K, svd_solver="randomized", random_state=0).fit(Xtr)
        clf = LogisticRegression(C=LR_C, max_iter=500, solver="lbfgs", tol=1e-4)
        clf.fit(pca.transform(Xtr), y[tr])
        accs[i] = clf.score(pca.transform(Xte), y[te])
    print(f"  {name:<10s}  D={X.shape[1]:>6d}  K={pca_k}  "
          f"mean={accs.mean():.3f} ± {accs.std():.3f}  "
          f"median={np.median(accs):.3f}  elapsed={time.time()-t0:.0f}s", flush=True)
    return accs


def main():
    pools, labels, subjects = build_features()
    m = labels >= 0
    print(f"\nLOSO 4-class probe on each pooling variant ({m.sum()} trials):", flush=True)
    results = {}
    for name in ["attention", "mean", "mean_chan", "mean_patch"]:
        X = pools[name][m]
        y = labels[m].astype(int)
        s = subjects[m]
        results[name] = loso_probe(X, y, s, PCA_K, name)

    lines = ["LOSO 4-class linear probe across pooling strategies:", ""]
    for name, accs in results.items():
        lines.append(f"  {name:<10s}: {accs.mean():.3f} ± {accs.std():.3f}  "
                     f"(median {np.median(accs):.3f})")
    lines.append("\n  paper reports 0.510 ± 0.012")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
