"""UMAP visualization of the K=50 reduced REVE latent space, colored by class.

Inputs cached features from train_subspace_probe.py:
  features (N_windows, 12850)  =  U_K(CLS, 50)  ||  V_K(tokens 256x50, 12800)
  labels   (N_windows,)        =  4-class motor imagery (0=L, 1=R, 2=B, 3=F)

Produces 3 UMAP scatter plots:
  (a) CLS only            — 50-D U_K coordinates
  (b) Tokens-mean only    — V_K coordinates averaged across the 256 tokens (50-D)
  (c) Full subspace       — 12850-D concatenation
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

FEAT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_subspace_features.npz")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_umap.pdf")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_umap_embeddings.npz")

K = 50
N_TOK = 256
CLASS_NAMES = ["L", "R", "B", "F"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
SEED = 42


def run_umap(X, label):
    print(f"  UMAP on {label}: shape={X.shape} ...", flush=True)
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED, verbose=False)
    emb = reducer.fit_transform(X)
    print(f"    done in {time.time()-t0:.0f}s", flush=True)
    return emb


def scatter_panel(ax, emb, labels, title):
    for c, name in enumerate(CLASS_NAMES):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], c=COLORS[c], label=name,
                   s=3, alpha=0.5, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", markerscale=3, fontsize=9)


def main():
    d = np.load(FEAT_NPZ)
    features = d["features"].astype(np.float32)
    labels = d["labels"].astype(np.int64)
    subjects = d["subjects"]
    n = features.shape[0]
    print(f"loaded {n} windows, feature dim {features.shape[1]}", flush=True)

    cls_feats = features[:, :K]                                      # (N, 50)
    tok_feats = features[:, K:].reshape(n, N_TOK, K).mean(axis=1)    # (N, 50)
    full_feats = features                                             # (N, 12850)

    emb_cls = run_umap(cls_feats, "CLS-only (50-D)")
    emb_tok = run_umap(tok_feats, "tokens-mean (50-D)")
    emb_full = run_umap(full_feats, "full subspace (12850-D)")

    np.savez_compressed(OUT_NPZ,
                        emb_cls=emb_cls.astype(np.float32),
                        emb_tok=emb_tok.astype(np.float32),
                        emb_full=emb_full.astype(np.float32),
                        labels=labels, subjects=subjects)
    print(f"saved embeddings -> {OUT_NPZ}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scatter_panel(axes[0], emb_cls, labels,
                  "CLS coords (U_K, 50-D)")
    scatter_panel(axes[1], emb_tok, labels,
                  "Token coords mean (V_K, 50-D)")
    scatter_panel(axes[2], emb_full, labels,
                  "Full subspace (12850-D)")
    fig.suptitle(
        f"UMAP of REVE K=50 main-subspace projections, colored by class  (N={n})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
