"""Per-patch grouped SVD on REVE latents.

For each window we build a (33280, 4) matrix E_w by stacking, for each patch p:
    column_p = [ cls_w (512); token_(ch=0, p) (512); ...; token_(ch=63, p) (512) ]
where cls_w is the per-window attention-pooled CLS using REVE's HF cls_query_token
(same vector across all 4 patches of that window).

We concatenate E_w across all motor-imagery windows along columns:
    X = [E_1 | E_2 | ... | E_N]  in R^{33280 x 4N}
and compute the eigenvalues of X Xᵀ (equivalently sigma_i^2 of the SVD of X).

Implemented via streaming randomized SVD (no materialization of X).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/media/alex/DATA1/REVE/hf_cache")
_TOK = Path("/media/alex/DATA1/REVE/.hf_token")
if _TOK.exists():
    os.environ.setdefault("HF_TOKEN", _TOK.read_text().strip())

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoModel

LAT_DIR = Path("/media/alex/DATA1/REVE/latents_paper")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_per_patch_svd.pdf")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_per_patch_svd.txt")

D = 512
C = 64
P = 4
ROW_DIM = (C + 1) * D                      # 33280
K_TOP = 100
OVERSAMPLE = 10
SEED = 0


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def build_subject_chunk(lat, q_cls, scale):
    """lat: (N, C, P, D) fp32. Returns Xs of shape (4*N, ROW_DIM): each row is a column of X."""
    N = lat.shape[0]
    flat = lat.reshape(N, C * P, D)
    scores = (flat @ q_cls) * scale
    w = softmax(scores, axis=1)
    cls_w = (w[..., None] * flat).sum(axis=1)        # (N, D)

    # Output rows in column order: patch-major (patch 0 then patch 1 ...). Within each patch,
    # one row per window. Each row: [cls_w (D); token_chan_0_patch_p (D); ... ; token_chan_63 (D)].
    Xs = np.empty((P * N, ROW_DIM), dtype=np.float32)
    for p in range(P):
        chans = lat[:, :, p, :].reshape(N, C * D)    # (N, 64*512)
        full = np.concatenate([cls_w, chans], axis=1) # (N, 33280)
        Xs[p * N:(p + 1) * N] = full
    return Xs


def stream_X_times_Omega(files, q_cls, scale, Omega):
    """Compute X @ Omega without materializing X.

    Omega: (total_cols, k). Returns (ROW_DIM, k).
    """
    Q_partial = np.zeros((ROW_DIM, Omega.shape[1]), dtype=np.float32)
    col_idx = 0
    for i, path in enumerate(files, 1):
        with h5py.File(path, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)
        N = lat.shape[0]
        Xs = build_subject_chunk(lat, q_cls, scale)              # (4N, ROW_DIM)
        block = Omega[col_idx:col_idx + P * N]                    # (4N, k)
        Q_partial += Xs.T @ block                                 # (ROW_DIM, k)
        col_idx += P * N
        if i % 20 == 0 or i == len(files):
            print(f"    pass1 {i}/{len(files)}", flush=True)
    return Q_partial, col_idx                                     # col_idx = total_cols


def stream_QT_times_X(files, q_cls, scale, Q, total_cols):
    """Compute B = Q.T @ X without materializing X. Returns (k, total_cols)."""
    B = np.zeros((Q.shape[1], total_cols), dtype=np.float32)
    col_idx = 0
    for i, path in enumerate(files, 1):
        with h5py.File(path, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)
        N = lat.shape[0]
        Xs = build_subject_chunk(lat, q_cls, scale)              # (4N, ROW_DIM)
        B[:, col_idx:col_idx + P * N] = Q.T @ Xs.T               # (k, 4N)
        col_idx += P * N
        if i % 20 == 0 or i == len(files):
            print(f"    pass2 {i}/{len(files)}", flush=True)
    return B


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q_cls = model.cls_query_token.detach().cpu().numpy().reshape(-1)
    scale = D ** -0.5
    del model

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    print(f"{len(files)} subjects", flush=True)

    # Count total columns (4 * total motor windows)
    total_cols = 0
    for path in files:
        with h5py.File(path, "r") as f:
            total_cols += P * f["trial/latent"].shape[0]
    print(f"X shape: ({ROW_DIM}, {total_cols})  -- {total_cols // P} windows", flush=True)

    rng = np.random.default_rng(SEED)
    n_random = K_TOP + OVERSAMPLE
    print(f"Generating Omega ({total_cols}, {n_random}) ...", flush=True)
    Omega = rng.standard_normal((total_cols, n_random)).astype(np.float32)

    print("Pass 1: streaming X @ Omega ...", flush=True)
    Q_partial, _ = stream_X_times_Omega(files, q_cls, scale, Omega)

    print("QR ...", flush=True)
    Q, _ = np.linalg.qr(Q_partial)                                # (ROW_DIM, n_random)

    print("Pass 2: streaming Q.T @ X ...", flush=True)
    B = stream_QT_times_X(files, q_cls, scale, Q, total_cols)     # (n_random, total_cols)

    print("Final SVD of small matrix ...", flush=True)
    U_B, sigma, Vt = np.linalg.svd(B, full_matrices=False)
    sigma = sigma[:K_TOP]                                          # top-K_TOP singular values
    eig = sigma ** 2                                               # eigvals of X X^T

    p = eig / eig.sum()
    p = p[p > 0]
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))

    cum = np.cumsum(eig) / eig.sum()
    k50 = int(np.searchsorted(cum, 0.5) + 1)
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)

    msg = (
        f"Per-patch grouped SVD (uncentered) of REVE latents (paper preprocessing)\n"
        f"  X shape   : ({ROW_DIM}, {total_cols})\n"
        f"  rows      : 1 CLS x 512 + 64 channels x 512 = {ROW_DIM}\n"
        f"  cols      : {total_cols // P} windows x 4 patches\n"
        f"  computed  : top-{K_TOP} singular values via streaming randomized SVD\n"
        f"  eff_rank  : {eff_rank:.2f}  (within top-{K_TOP})\n"
        f"  k50={k50}   k90={k90}   k99={k99}\n"
        f"  log10 sigma^2 top-10: {np.log10(np.maximum(eig[:10], 1e-30)).round(2).tolist()}\n"
        f"  log10 sigma^2 around 6-10: {np.log10(np.maximum(eig[5:11], 1e-30)).round(2).tolist()}\n"
    )
    print()
    print(msg)
    OUT_TXT.write_text(msg)

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].semilogy(eig, "-o", markersize=4)
        axes[0].set_title(
            rf"$\sigma_i^2$ (log) top-{K_TOP}    "
            rf"eff_rank={eff_rank:.2f}    k90={k90}   k99={k99}"
        )
        axes[0].set_xlabel("rank"); axes[0].set_ylabel(r"$\sigma_i^2$ (log)")
        axes[0].grid(True, which="both", alpha=0.3)

        axes[1].plot(100 * cum, "-o", markersize=4)
        axes[1].axhline(50, color="gray", linestyle=":")
        axes[1].axhline(90, color="gray", linestyle=":")
        axes[1].axhline(99, color="gray", linestyle=":")
        axes[1].set_title("cumulative variance %")
        axes[1].set_xlabel("rank"); axes[1].set_ylabel("cum %")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(
            f"Per-patch grouped SVD (uncentered): X = ({ROW_DIM}, {total_cols})",
            fontsize=11,
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
