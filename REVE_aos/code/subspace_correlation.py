"""Subspace correlation between leading eigenvector subspaces of REVE token covariances.

Hypothesis: top-few directions (the architecture-determined "outlier features") are
nearly identical across subjects -> very high subspace correlation. The next chunk
(directions ~6..18 that bring effective rank from ~3 up to ~13 in the residual)
might carry input-dependent structure and show lower across-subject alignment.

For each subject we compute the top-K_MAX eigenvectors of:
- "grand"          : pooled covariance over all motor-class trial tokens (no rest)
- "L", "R", "B", "F": per-class covariances

Subspace correlation between two orthonormal bases U, V (each D x k):
    rho(U, V) = ||U^T V||_F^2 / k       in [0, 1]
    = 1 iff identical spans
    = 0 iff orthogonal

Outputs (in /media/alex/DATA1/REVE/reports/):
- reve_subspace_correlation.pdf   -- 4 pages of figures
- reve_subspace_correlation.npz   -- raw arrays for downstream use
"""
from __future__ import annotations

import glob
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LAT_DIR = Path("/media/alex/DATA1/REVE/latents")
OUT_DIR = Path("/media/alex/DATA1/REVE/reports")
OUT_PDF = OUT_DIR / "reve_subspace_correlation.pdf"
OUT_NPZ = OUT_DIR / "reve_subspace_correlation.npz"

CLASSES        = [(0, "L"), (1, "R"), (2, "B"), (3, "F")]
CLASS_NAMES    = [name for _, name in CLASSES]
D              = 512
K_MAX          = 50                                # max subspace dim to evaluate
K_GRID_HM      = [5, 10, 15, 20, 30, 50]           # k values for heatmaps
K_GRID_CURVE   = list(range(2, K_MAX + 1))         # k values for curves


# ---------- helpers ----------
def cov_from_tokens(toks: np.ndarray) -> np.ndarray:
    n = toks.shape[0]
    Xc = toks - toks.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    return (cov + cov.T) * 0.5


def top_k_eigvecs(cov: np.ndarray, k: int) -> np.ndarray:
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx].astype(np.float32)


def cumulative_corr(U: np.ndarray, V: np.ndarray, k_grid) -> np.ndarray:
    """For each k in k_grid, return ||U[:,:k]^T V[:,:k]||_F^2 / k."""
    Kmax = max(k_grid)
    G = U[:, :Kmax].T @ V[:, :Kmax]                # (Kmax, Kmax)
    G2 = G ** 2
    return np.array([G2[:k, :k].sum() / k for k in k_grid], dtype=np.float32)


# ---------- main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    sids = [int(Path(p).stem[1:]) for p in files]
    n_subj = len(files)
    print(f"{n_subj} subjects")

    print("Computing eigenvectors (top-{}) per subject and class ...".format(K_MAX))
    eigvecs = []                                   # list of dicts
    for i, p in enumerate(files):
        with h5py.File(p, "r") as f:
            lat = f["windows/trial/latent"][:].astype(np.float32)  # (N, C, P, D)
            lbl = f["windows/trial/label"][:]
        sub = {}
        # grand: pool all motor-class trial tokens (exclude rest)
        m = lbl >= 0
        if m.any():
            toks = lat[m].reshape(-1, D).astype(np.float64)
            sub["grand"] = top_k_eigvecs(cov_from_tokens(toks), K_MAX)
        else:
            sub["grand"] = None
        for c, name in CLASSES:
            mask = lbl == c
            if not mask.any():
                sub[name] = None; continue
            toks = lat[mask].reshape(-1, D).astype(np.float64)
            sub[name] = top_k_eigvecs(cov_from_tokens(toks), K_MAX)
        eigvecs.append(sub)
        if (i + 1) % 20 == 0 or (i + 1) == n_subj:
            print(f"  {i+1}/{n_subj}")

    # 1. Across-subject grand-subspace correlation matrices for k in K_GRID_HM
    print("Across-subject pairwise correlations (grand subspace) ...")
    n_curve = len(K_GRID_CURVE)
    pair_idx_to_ij = list(combinations(range(n_subj), 2))
    n_pairs = len(pair_idx_to_ij)

    # Compute curve-grid values per pair (one (n_curve,) vector per pair)
    pair_curves = np.empty((n_pairs, n_curve), dtype=np.float32)
    for p_idx, (i, j) in enumerate(pair_idx_to_ij):
        pair_curves[p_idx] = cumulative_corr(
            eigvecs[i]["grand"], eigvecs[j]["grand"], K_GRID_CURVE
        )
        if (p_idx + 1) % 1000 == 0 or (p_idx + 1) == n_pairs:
            print(f"  pair {p_idx+1}/{n_pairs}")

    # Build heatmaps for the K_GRID_HM subset by indexing into the curve
    K_GRID_CURVE_arr = np.array(K_GRID_CURVE)
    heatmaps = {}
    for k in K_GRID_HM:
        col = int(np.where(K_GRID_CURVE_arr == k)[0][0])
        H = np.eye(n_subj, dtype=np.float32)
        for p_idx, (i, j) in enumerate(pair_idx_to_ij):
            v = pair_curves[p_idx, col]
            H[i, j] = v; H[j, i] = v
        heatmaps[k] = H

    mean_offdiag = pair_curves.mean(axis=0)        # (n_curve,)
    p10_offdiag  = np.percentile(pair_curves, 10, axis=0)
    p90_offdiag  = np.percentile(pair_curves, 90, axis=0)

    # 2. Within-subject across-class alignment (mean over 6 pairs of {L,R,B,F})
    print("Within-subject across-class correlations ...")
    within_class = np.full((n_subj, n_curve), np.nan, dtype=np.float32)
    for s in range(n_subj):
        ev = eigvecs[s]
        Us = [ev[name] for name in CLASS_NAMES if ev[name] is not None]
        if len(Us) < 2:
            continue
        vals = np.stack([cumulative_corr(a, b, K_GRID_CURVE) for a, b in combinations(Us, 2)])
        within_class[s] = vals.mean(axis=0)

    # 3. Per-class subspace vs grand subspace within subject
    class_vs_grand = np.full((n_subj, 4, n_curve), np.nan, dtype=np.float32)
    for s in range(n_subj):
        ev = eigvecs[s]
        Ug = ev["grand"]
        if Ug is None:
            continue
        for ci, (_, name) in enumerate(CLASSES):
            U = ev[name]
            if U is None:
                continue
            class_vs_grand[s, ci] = cumulative_corr(Ug, U, K_GRID_CURVE)

    # ---------- render ----------
    print("Rendering PDF ...")
    with PdfPages(OUT_PDF) as pdf:
        # Page 1: heatmap grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        last_im = None
        for ax, k in zip(axes.flat, K_GRID_HM):
            H = heatmaps[k]
            offdiag = H[np.triu_indices(n_subj, 1)]
            last_im = ax.imshow(H, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(
                f"k={k}    mean off-diag={offdiag.mean():.3f}    min={offdiag.min():.3f}",
                fontsize=10,
            )
            ax.set_xlabel("subject"); ax.set_ylabel("subject")
        fig.suptitle(
            "Across-subject grand-subspace correlation\n"
            r"$\rho = \|U_k^T V_k\|_F^2 / k$  in [0, 1]   (108 subjects, 5778 pairs)",
            fontsize=11,
        )
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.7)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: mean off-diag vs k
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(K_GRID_CURVE, p10_offdiag, p90_offdiag, alpha=0.25,
                        color="C0", label="10–90 pct")
        ax.plot(K_GRID_CURVE, mean_offdiag, "-o", color="C0", markersize=3, label="mean")
        ax.set_xlabel("subspace dimension k")
        ax.set_ylabel("across-subject subspace correlation")
        ax.set_title("Across-subject grand-subspace alignment vs k  (5778 pairs)")
        ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3); ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: within-subject across-class alignment
        fig, ax = plt.subplots(figsize=(10, 5))
        for s in range(n_subj):
            ax.plot(K_GRID_CURVE, within_class[s], color="lightblue",
                    alpha=0.35, linewidth=0.6)
        ax.plot(K_GRID_CURVE, np.nanmean(within_class, axis=0), color="navy",
                linewidth=2, label="mean across 108 subjects")
        ax.set_xlabel("subspace dimension k")
        ax.set_ylabel("within-subject across-class subspace correlation")
        ax.set_title("Within-subject: alignment among L / R / B / F subspaces vs k\n"
                     "(thin: per-subject mean of 6 class pairs; bold: across-subject mean)")
        ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3); ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: per-class vs grand within subject
        fig, ax = plt.subplots(figsize=(10, 5))
        means = np.nanmean(class_vs_grand, axis=0)  # (4, n_curve)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for ci, (_, name) in enumerate(CLASSES):
            ax.plot(K_GRID_CURVE, means[ci], "-o", markersize=3, color=colors[ci], label=name)
        ax.set_xlabel("subspace dimension k")
        ax.set_ylabel("class subspace ↔ grand subspace correlation (mean across 108 subj)")
        ax.set_title("Within-subject: alignment of each class subspace to the grand subspace")
        ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3); ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    np.savez_compressed(
        OUT_NPZ,
        K_GRID_HM=np.array(K_GRID_HM),
        K_GRID_CURVE=np.array(K_GRID_CURVE),
        mean_offdiag=mean_offdiag,
        p10_offdiag=p10_offdiag,
        p90_offdiag=p90_offdiag,
        within_class=within_class,
        class_vs_grand=class_vs_grand,
        **{f"H_k{k}": h for k, h in heatmaps.items()},
        subject_ids=np.array(sids),
    )

    print(f"\nwrote {OUT_PDF}")
    print(f"wrote {OUT_NPZ}")
    print()
    print("across-subject grand-subspace alignment:")
    for k in K_GRID_HM:
        col = int(np.where(K_GRID_CURVE_arr == k)[0][0])
        print(f"  k={k:2d}: mean={mean_offdiag[col]:.3f}  "
              f"p10={p10_offdiag[col]:.3f}  p90={p90_offdiag[col]:.3f}")
    print("within-subject across-class alignment:")
    for k in K_GRID_HM:
        col = int(np.where(K_GRID_CURVE_arr == k)[0][0])
        x = within_class[:, col]; x = x[np.isfinite(x)]
        print(f"  k={k:2d}: mean={x.mean():.3f}  min={x.min():.3f}  max={x.max():.3f}")


if __name__ == "__main__":
    main()
