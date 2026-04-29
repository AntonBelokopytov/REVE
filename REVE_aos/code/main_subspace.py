"""Find REVE's main per-token subspace via UNCENTERED eigendecomposition.

X ∈ R^{N x 512} stacks every (window, channel, patch) token as a row.
M = X^T X / N    (no mean subtraction — second-moment / "uncorrected covariance")
Eigendecompose M to get eigenvalues lambda_i and eigenvectors V (512 x 512).
Top-K eigenvectors V_K span the "main subspace"; main coordinates of any token x
are V_K^T x.

Reads /media/alex/DATA1/REVE/latents_paper/ (paper-preprocessed extraction).

Outputs:
  reve_main_subspace.npz   (V_K, eigvals, mu, K, N_tokens)
  reve_main_subspace.pdf   (spectrum + V_K heatmap + per-coord max loading)
  reve_main_subspace.txt   (rank stats + top coord indices)
"""
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LAT_DIR = Path("/media/alex/DATA1/REVE/latents_paper")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_main_subspace.npz")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_main_subspace.pdf")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_main_subspace.txt")

D = 512
K_SAVE = 50           # how many top eigvecs to save in V_K
THRESHOLDS = [0.10, 0.15, 0.20, 0.30]


def participation_ratio(v):
    s4 = (v ** 4).sum()
    return float(1.0 / s4) if s4 > 0 else 0.0


def main():
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    print(f"{len(files)} subjects", flush=True)

    n_tokens = 0
    sx = np.zeros(D, dtype=np.float64)              # mean tracker (for reference)
    M = np.zeros((D, D), dtype=np.float64)          # uncentered second moment

    for i, p in enumerate(files, 1):
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)   # (N, C, P, D)
        toks = lat.reshape(-1, D).astype(np.float64)
        n_tokens += toks.shape[0]
        sx += toks.sum(axis=0)
        M += toks.T @ toks
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  total tokens: {n_tokens:,}", flush=True)

    M /= n_tokens
    M = (M + M.T) * 0.5
    mu = sx / n_tokens                              # global mean (kept for reference; not subtracted)

    # eigendecompose
    w, V = np.linalg.eigh(M)
    order = np.argsort(w)[::-1]
    eigvals = w[order]
    V = V[:, order]
    V_K = V[:, :K_SAVE].astype(np.float32)

    # rank stats
    e = np.maximum(eigvals, 0.0)
    s = e.sum()
    p = e[e > 0] / s
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    cum = np.cumsum(e) / s
    k50 = int(np.searchsorted(cum, 0.5) + 1)
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)

    # active coords from top-5 PCs
    top5 = V_K[:, :5]
    row_max5 = np.abs(top5).max(axis=1)
    actives = {}
    for thr in THRESHOLDS:
        u = set()
        for k in range(5):
            u |= set(np.where(np.abs(top5[:, k]) > thr)[0].tolist())
        actives[thr] = sorted(u)

    # Participation ratio per PC (sparsity diagnostic)
    pr = [participation_ratio(V_K[:, k]) for k in range(min(10, K_SAVE))]

    lines = []
    lines.append("Main subspace via uncentered eigendecomposition of per-token tokens")
    lines.append(f"  source : {LAT_DIR}")
    lines.append(f"  N tokens = {n_tokens:,}, dim = {D}")
    lines.append(f"  M = X^T X / N (no mean subtraction)")
    lines.append("")
    lines.append(f"  effective rank = {eff_rank:.2f}")
    lines.append(f"  k50 = {k50}   k90 = {k90}   k99 = {k99}")
    lines.append(f"  top-10 eigvals (log10): "
                 f"{np.log10(np.maximum(eigvals[:10], 1e-30)).round(2).tolist()}")
    lines.append(f"  top-10 eigvals       : {eigvals[:10].round(3).tolist()}")
    lines.append("")
    lines.append("  participation ratio (effective # of coords) per PC:")
    for k, r in enumerate(pr, 1):
        lines.append(f"    PC{k:2d}: {r:6.2f} eff coords   eigval = {eigvals[k-1]:.3g}")
    lines.append("")
    lines.append("  active coords union over top-5 PCs:")
    for thr, idx in actives.items():
        lines.append(f"    |V|>{thr}: {len(idx)} coords  -> {idx[:15]}{'...' if len(idx)>15 else ''}")
    lines.append("")
    lines.append(f"  NOTE: global mean is non-zero in uncentered analysis. ||mu||_2 = {np.linalg.norm(mu):.3g}")
    lines.append(f"        For comparison, sqrt(eigval[0]) = {np.sqrt(eigvals[0]):.3g}")
    lines.append("")
    msg = "\n".join(lines)
    print()
    print(msg)
    OUT_TXT.write_text(msg)

    np.savez_compressed(
        OUT_NPZ,
        V_K=V_K,
        eigvals=eigvals.astype(np.float32),
        mu=mu.astype(np.float32),
        K=np.int64(K_SAVE),
        n_tokens=np.int64(n_tokens),
        active_coords_thr10=np.array(actives[0.10], dtype=np.int32),
        active_coords_thr15=np.array(actives[0.15], dtype=np.int32),
    )
    print(f"\nwrote {OUT_NPZ}")
    print(f"wrote {OUT_TXT}")

    # PDF
    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].semilogy(np.maximum(eigvals, 1e-30), '-o', markersize=3)
        axes[0].set_xlabel("rank")
        axes[0].set_ylabel("eigenvalue (log)")
        axes[0].set_title(f"spectrum  eff_rank={eff_rank:.2f}\nk50={k50}  k90={k90}  k99={k99}")
        axes[0].grid(True, which="both", alpha=0.3)

        # V_K heatmap (signed) for top-50 PCs
        vmax = float(np.percentile(np.abs(V_K), 99.5))
        im = axes[1].imshow(V_K, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                            interpolation="nearest", aspect="auto")
        axes[1].set_xlabel("PC (1..K)")
        axes[1].set_ylabel("embedding coord (0..511)")
        axes[1].set_title(f"V_K (signed)  K={K_SAVE}  clip ±{vmax:.3f}")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # per-coord max loading among top-5 PCs
        axes[2].bar(np.arange(D), row_max5, width=1.0, color="steelblue")
        axes[2].set_xlabel("embedding coord (0..511)")
        axes[2].set_ylabel("max |V| over top-5 PCs")
        axes[2].set_title(f"per-coord top-5 loading\n"
                          f"{(row_max5>0.1).sum()} coords > 0.1")
        for idx in np.argsort(row_max5)[::-1][:8]:
            axes[2].text(idx, row_max5[idx] + 0.012, str(idx),
                         ha="center", fontsize=7, rotation=90)
        fig.suptitle(
            f"Main per-token subspace (uncentered eigendecomp, N={n_tokens:,})",
            fontsize=11,
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
