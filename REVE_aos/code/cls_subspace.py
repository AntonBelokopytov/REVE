"""Find the main subspace U_K of the per-window CLS token (attention-pooled).

For each window, the CLS is the attention-pooled summary computed using REVE's HF
cls_query_token applied to the (C*P, D) tokens of that window.

Y ∈ R^{N_windows x 512} stacks one CLS per window.
M_cls = Y^T Y / N_windows  (uncentered, no mean subtraction).
Eigendecompose M_cls to get U (512x512); top-K is U_K.

Outputs:
  reve_cls_subspace.npz    (U_K, eigvals, mu_cls, K, n_windows)
  reve_cls_subspace.pdf    (spectrum + U_K heatmap + per-coord max loading)
  reve_cls_subspace.txt
"""


"""
check

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
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoModel

LAT_DIR = Path("/media/alex/DATA1/REVE/latents_paper")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_cls_subspace.npz")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_cls_subspace.pdf")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_cls_subspace.txt")

D = 512
K_SAVE = 50
THRESHOLDS = [0.10, 0.15, 0.20, 0.30]


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def participation_ratio(v):
    s4 = (v ** 4).sum()
    return float(1.0 / s4) if s4 > 0 else 0.0


def main():
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(-1)
    scale = D ** -0.5
    del model

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    print(f"{len(files)} subjects", flush=True)

    cls_all = []
    for i, p in enumerate(files, 1):
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)        # (N, C, P, D)
        N, C, P, _ = lat.shape
        flat = lat.reshape(N, C * P, D)
        scores = (flat @ q) * scale
        w = softmax(scores, axis=1)
        cls = (w[..., None] * flat).sum(axis=1).astype(np.float64)   # (N, D)
        cls_all.append(cls)
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    Y = np.concatenate(cls_all, axis=0)
    n_windows = Y.shape[0]
    print(f"Y shape: {Y.shape}", flush=True)

    mu_cls = Y.mean(axis=0)
    M = (Y.T @ Y) / n_windows
    M = (M + M.T) * 0.5

    w_, U = np.linalg.eigh(M)
    order = np.argsort(w_)[::-1]
    eigvals = w_[order]
    U = U[:, order]
    U_K = U[:, :K_SAVE].astype(np.float32)

    e = np.maximum(eigvals, 0.0); s = e.sum()
    p = e[e > 0] / s
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    cum = np.cumsum(e) / s
    k50 = int(np.searchsorted(cum, 0.5) + 1)
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)

    top5 = U_K[:, :5]
    row_max5 = np.abs(top5).max(axis=1)
    actives = {}
    for thr in THRESHOLDS:
        u = set()
        for k in range(5):
            u |= set(np.where(np.abs(top5[:, k]) > thr)[0].tolist())
        actives[thr] = sorted(u)

    pr = [participation_ratio(U_K[:, k]) for k in range(min(10, K_SAVE))]

    # compare U_K to V_K (per-token main subspace) by subspace correlation
    v_path = OUT_NPZ.parent / "reve_main_subspace.npz"
    cmp_lines = []
    if v_path.exists():
        v = np.load(v_path)["V_K"]
        for k in [5, 10, 20, 50]:
            kk = min(k, U_K.shape[1], v.shape[1])
            G = U_K[:, :kk].T @ v[:, :kk]
            rho = float((G ** 2).sum() / kk)
            cmp_lines.append(f"  ||U_{kk}^T V_{kk}||_F^2 / {kk}  =  {rho:.3f}")
    cmp_block = "\n".join(cmp_lines) if cmp_lines else "  (V_K not found)"

    lines = []
    lines.append("CLS subspace via uncentered eigendecomposition of per-window CLS tokens")
    lines.append(f"  source : {LAT_DIR}")
    lines.append(f"  N windows = {n_windows:,}, dim = {D}")
    lines.append(f"  CLS = REVE attention pool with HF cls_query_token (per window)")
    lines.append(f"  M = Y^T Y / N (no mean subtraction)")
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
    lines.append("  active coords union over top-5 PCs of U_K:")
    for thr, idx in actives.items():
        lines.append(f"    |U|>{thr}: {len(idx)} coords  -> {idx[:15]}{'...' if len(idx)>15 else ''}")
    lines.append("")
    lines.append(f"  ||mu_cls||_2 = {np.linalg.norm(mu_cls):.3g}   sqrt(eigval[0]) = {np.sqrt(eigvals[0]):.3g}")
    lines.append("")
    lines.append("  Subspace correlation between CLS subspace U_K and per-token V_K:")
    lines.append(cmp_block)
    msg = "\n".join(lines)
    print()
    print(msg)
    OUT_TXT.write_text(msg)

    np.savez_compressed(
        OUT_NPZ,
        U_K=U_K,
        eigvals=eigvals.astype(np.float32),
        mu_cls=mu_cls.astype(np.float32),
        K=np.int64(K_SAVE),
        n_windows=np.int64(n_windows),
        active_coords_thr10=np.array(actives[0.10], dtype=np.int32),
        active_coords_thr15=np.array(actives[0.15], dtype=np.int32),
    )
    print(f"\nwrote {OUT_NPZ}")
    print(f"wrote {OUT_TXT}")

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].semilogy(np.maximum(eigvals, 1e-30), "-o", markersize=3)
        axes[0].set_xlabel("rank"); axes[0].set_ylabel("eigenvalue (log)")
        axes[0].set_title(f"CLS spectrum  eff_rank={eff_rank:.2f}\nk50={k50}  k90={k90}  k99={k99}")
        axes[0].grid(True, which="both", alpha=0.3)

        vmax = float(np.percentile(np.abs(U_K), 99.5))
        im = axes[1].imshow(U_K, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                            interpolation="nearest", aspect="auto")
        axes[1].set_xlabel("PC"); axes[1].set_ylabel("embedding coord (0..511)")
        axes[1].set_title(f"U_K (signed)  K={K_SAVE}  clip ±{vmax:.3f}")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].bar(np.arange(D), row_max5, width=1.0, color="darkorange")
        axes[2].set_xlabel("embedding coord (0..511)")
        axes[2].set_ylabel("max |U| over top-5 PCs")
        axes[2].set_title(f"per-coord top-5 loading\n{(row_max5>0.1).sum()} coords > 0.1")
        for idx in np.argsort(row_max5)[::-1][:8]:
            axes[2].text(idx, row_max5[idx] + 0.012, str(idx),
                         ha="center", fontsize=7, rotation=90)
        fig.suptitle(
            f"CLS-token main subspace (uncentered, N={n_windows:,} windows)",
            fontsize=11,
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
