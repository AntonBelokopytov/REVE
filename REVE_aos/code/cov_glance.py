"""Compute 512x512 covariances of REVE token embeddings and dump heatmaps + spectra to a PDF.

Each REVE per-trial latent is (channels, patches, 512). We treat each (channel, patch)
token as one 512-d sample. Covariance is then the population covariance of those tokens.

Outputs:
  /media/alex/DATA1/REVE/reports/reve_token_covariance.pdf  -- heatmaps + spectra
  /media/alex/DATA1/REVE/reports/reve_token_covariance.npz  -- cached cov matrices
"""
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LAT_DIR  = Path("/media/alex/DATA1/REVE/latents")
OUT_DIR  = Path("/media/alex/DATA1/REVE/reports")
OUT_PDF  = OUT_DIR / "reve_token_covariance.pdf"
OUT_NPZ  = OUT_DIR / "reve_token_covariance.npz"

CLASSES = [(-1, "rest"), (0, "L"), (1, "R"), (2, "B"), (3, "F")]
D = 512


def effective_rank(eigvals: np.ndarray) -> float:
    """exp(entropy of normalized eigenvalues) — number of dims to spread variance evenly."""
    e = np.maximum(eigvals, 0.0)
    s = e.sum()
    if s <= 0:
        return 0.0
    p = e / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def render_page(pdf: PdfPages, title: str, cov: np.ndarray,
                corr: np.ndarray, eigvals: np.ndarray) -> None:
    fig = plt.figure(figsize=(15, 4.6))

    ax = fig.add_subplot(1, 3, 1)
    vmax = np.percentile(np.abs(cov), 99)
    im = ax.imshow(cov, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{title}\ncovariance (clip {vmax:.2g})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 3, 2)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title("correlation")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 3, 3)
    eff = effective_rank(eigvals)
    cum = np.cumsum(eigvals) / eigvals.sum()
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    ax.semilogy(np.maximum(eigvals, 1e-12))
    ax.set_xlabel("component rank")
    ax.set_ylabel("eigenvalue (log)")
    ax.set_title(f"spectrum  eff_rank={eff:.1f}  k90={k90}  k99={k99}")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def cov_from_sums(n: int, sx: np.ndarray, sxx: np.ndarray) -> np.ndarray:
    mu = sx / n
    cov = (sxx - n * np.outer(mu, mu)) / (n - 1)
    return (cov + cov.T) * 0.5


def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    sd[sd < 1e-12] = 1.0
    return cov / np.outer(sd, sd)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    print(f"{len(files)} subjects")

    # Grand accumulators (over all tokens) and per-class accumulators
    n_all = 0
    sx_all  = np.zeros(D, dtype=np.float64)
    sxx_all = np.zeros((D, D), dtype=np.float64)

    n_c   = {c: 0 for c, _ in CLASSES}
    sx_c  = {c: np.zeros(D, dtype=np.float64) for c, _ in CLASSES}
    sxx_c = {c: np.zeros((D, D), dtype=np.float64) for c, _ in CLASSES}

    for i, p in enumerate(files, 1):
        with h5py.File(p, "r") as f:
            lat = f["windows/trial/latent"][:].astype(np.float32)   # (N, C, P, D)
            lbl = f["windows/trial/label"][:]                        # (N,)
        N, C, P, _ = lat.shape
        toks = lat.reshape(-1, D)
        tok_lbl = np.repeat(lbl, C * P)

        n_all  += toks.shape[0]
        sx_all  += toks.sum(axis=0, dtype=np.float64)
        sxx_all += toks.astype(np.float64).T @ toks.astype(np.float64)

        for c, _ in CLASSES:
            m = tok_lbl == c
            if not m.any():
                continue
            sub = toks[m].astype(np.float64)
            n_c[c]   += sub.shape[0]
            sx_c[c]  += sub.sum(axis=0)
            sxx_c[c] += sub.T @ sub

        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)}  total tokens so far: {n_all:,}")

    cov_all  = cov_from_sums(n_all, sx_all, sxx_all)
    corr_all = corr_from_cov(cov_all)
    eig_all  = np.linalg.eigvalsh(cov_all)[::-1]

    cov_by_class  = {}
    corr_by_class = {}
    eig_by_class  = {}
    for c, name in CLASSES:
        if n_c[c] < 100:
            continue
        cov_c = cov_from_sums(n_c[c], sx_c[c], sxx_c[c])
        cov_by_class[c]  = cov_c
        corr_by_class[c] = corr_from_cov(cov_c)
        eig_by_class[c]  = np.linalg.eigvalsh(cov_c)[::-1]

    np.savez_compressed(
        OUT_NPZ,
        cov_all=cov_all, eig_all=eig_all, n_all=n_all,
        **{f"cov_{name}": cov_by_class[c] for c, name in CLASSES if c in cov_by_class},
        **{f"eig_{name}": eig_by_class[c] for c, name in CLASSES if c in eig_by_class},
        **{f"n_{name}":   np.int64(n_c[c]) for c, name in CLASSES if c in cov_by_class},
    )

    with PdfPages(OUT_PDF) as pdf:
        render_page(pdf, f"all tokens (N={n_all:,})", cov_all, corr_all, eig_all)
        for c, name in CLASSES:
            if c not in cov_by_class:
                continue
            render_page(
                pdf,
                f"class {name} (N={n_c[c]:,})",
                cov_by_class[c], corr_by_class[c], eig_by_class[c],
            )

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
