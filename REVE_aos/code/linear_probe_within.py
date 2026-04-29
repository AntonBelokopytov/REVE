"""Within-subject 5-fold CV linear probe on REVE per-window embeddings.

Hypothesis: the paper's 0.510 ± 0.012 comes from *within-subject* CV, since the
small std is incompatible with cross-subject LOSO on eegmmidb. We test that here.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

NPZ = Path("/media/alex/DATA1/REVE/reports/reve_pertrial_embeddings.npz")
OUT = Path("/media/alex/DATA1/REVE/reports/reve_linear_probe_within.txt")
PCA_K = 50
N_FOLDS = 5
LR_C = 1.0


def main():
    d = np.load(NPZ)
    emb = d["embeddings"].astype(np.float32)
    lbl = d["labels"]
    sub = d["subjects"]

    m = lbl >= 0
    X, y, s = emb[m], lbl[m].astype(int), sub[m]
    subjects = np.unique(s)
    print(f"{len(subjects)} subjects, {len(y)} motor trials", flush=True)

    accs = np.zeros(len(subjects))
    t0 = time.time()
    for i, sid in enumerate(subjects):
        idx = np.where(s == sid)[0]
        Xs, ys = X[idx], y[idx]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        fold_accs = []
        for tr_i, te_i in skf.split(Xs, ys):
            sc = StandardScaler().fit(Xs[tr_i])
            Xtr = sc.transform(Xs[tr_i])
            Xte = sc.transform(Xs[te_i])
            pca = PCA(n_components=min(PCA_K, len(tr_i) - 1),
                      svd_solver="randomized", random_state=0).fit(Xtr)
            Ztr = pca.transform(Xtr)
            Zte = pca.transform(Xte)
            clf = LogisticRegression(C=LR_C, max_iter=500, solver="lbfgs", tol=1e-4)
            clf.fit(Ztr, ys[tr_i])
            fold_accs.append(clf.score(Zte, ys[te_i]))
        accs[i] = np.mean(fold_accs)
        if i < 3 or (i + 1) % 20 == 0 or i == len(subjects) - 1:
            print(f"  subj {i+1:3d}/{len(subjects)} S{sid:03d}: "
                  f"acc={accs[i]:.3f}  running mean={accs[:i+1].mean():.3f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    msg = (
        f"\nWithin-subject 5-fold CV linear probe on per-window REVE embeddings:\n"
        f"  N subjects = {len(accs)}\n"
        f"  features   = top-{PCA_K} PCA components (per-fold)\n"
        f"  mean acc   = {accs.mean():.3f}\n"
        f"  std  acc   = {accs.std():.3f}\n"
        f"  median     = {np.median(accs):.3f}\n"
        f"  min/max    = {accs.min():.3f} / {accs.max():.3f}\n"
        f"  paper reports 0.510 ± 0.012\n"
    )
    print(msg)
    OUT.write_text(msg)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
