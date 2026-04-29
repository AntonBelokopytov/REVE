"""Leave-One-Subject-Out linear probe on REVE per-window embeddings.

Targets the paper's reported 0.510 ± 0.012 on 4-class motor imagery (eegmmidb).
Per-window REVE embeddings have effective rank ~6 in a 512-D ambient space, which
makes LBFGS on the full-D features pathologically slow. We project to top-K PCA
components (K=50 -> ~99% variance) inside each fold, train the linear classifier
on the projection, and report LOSO accuracy.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NPZ = Path("/media/alex/DATA1/REVE/reports/reve_pertrial_embeddings.npz")
OUT = Path("/media/alex/DATA1/REVE/reports/reve_linear_probe.txt")
PCA_K = 50          # top components, ~99% variance
LR_C  = 1.0


def main():
    d = np.load(NPZ)
    emb = d["embeddings"].astype(np.float32)
    lbl = d["labels"]
    sub = d["subjects"]

    m = lbl >= 0
    X, y, s = emb[m], lbl[m].astype(int), sub[m]
    print(f"motor trials: {len(y)}, subjects: {len(np.unique(s))}, "
          f"classes={dict(zip(*np.unique(y, return_counts=True)))}", flush=True)

    subjects = np.unique(s)
    accs = np.zeros(len(subjects))
    t0 = time.time()
    for i, sid in enumerate(subjects):
        tr = s != sid
        te = s == sid
        # standardize -> PCA-K -> LR, all fitted on train fold only
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xte = sc.transform(X[te])
        pca = PCA(n_components=PCA_K, svd_solver="randomized", random_state=0).fit(Xtr)
        Ztr = pca.transform(Xtr)
        Zte = pca.transform(Xte)
        clf = LogisticRegression(C=LR_C, max_iter=500, solver="lbfgs", tol=1e-4)
        clf.fit(Ztr, y[tr])
        accs[i] = clf.score(Zte, y[te])
        if i < 3 or (i + 1) % 20 == 0 or i == len(subjects) - 1:
            print(f"  fold {i+1:3d}/{len(subjects)}  S{sid:03d}  "
                  f"acc={accs[i]:.3f}  running mean={accs[:i+1].mean():.3f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    msg = (
        f"\nLOSO 4-class linear probe on per-window REVE embeddings:\n"
        f"  N folds (subjects) = {len(accs)}\n"
        f"  features = top-{PCA_K} PCA components of 512-D embedding (per-fold PCA)\n"
        f"  mean acc = {accs.mean():.3f}\n"
        f"  std  acc = {accs.std():.3f}\n"
        f"  median   = {np.median(accs):.3f}\n"
        f"  min/max  = {accs.min():.3f} / {accs.max():.3f}\n"
        f"  paper reports 0.510 ± 0.012\n"
    )
    print(msg)
    OUT.write_text(msg)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
