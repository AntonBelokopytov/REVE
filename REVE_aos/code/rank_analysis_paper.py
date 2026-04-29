"""Rank/spectrum analysis on the paper-preprocessed REVE latents.

Re-runs the same diagnostics we did on the earlier (volts-scaled, no preprocessing)
extraction, but on /media/alex/DATA1/REVE/latents_paper/ — which uses CAR + 0.3 Hz HP
+ 60 Hz notch + uV/100 and matches the protocol that reproduces the paper's number.

Reports:
  per-token: grand cov + spectrum, per-class spectra, active coords (top-5 PC support)
  per-window: attention-pooled (paper's frozen cls_query_token) cov + spectrum
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
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_rank_paper.pdf")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_rank_paper.txt")

D = 512
CLASSES = [(0, "L"), (1, "R"), (2, "B"), (3, "F")]


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def effective_rank(eig):
    e = np.maximum(eig, 0); s = e.sum()
    if s <= 0: return 0.0
    p = e / s; p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def k_at(eig, frac):
    e = np.maximum(eig, 0); s = e.sum()
    if s <= 0: return 0
    cum = np.cumsum(e) / s
    return int(np.searchsorted(cum, frac) + 1)


def cov_from_sums(n, sx, sxx):
    mu = sx / n
    cov = (sxx - n * np.outer(mu, mu)) / max(n - 1, 1)
    return (cov + cov.T) * 0.5


def participation_ratio(v):
    """1 / sum(v_i^4)  -- effective # of coords."""
    s4 = (v ** 4).sum()
    return float(1.0 / s4) if s4 > 0 else 0.0


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    # ---------- streaming aggregation over all trials/tokens ----------
    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    print(f"{len(files)} subjects", flush=True)

    n_tok_all = 0
    sx_tok = np.zeros(D, dtype=np.float64)
    sxx_tok = np.zeros((D, D), dtype=np.float64)
    n_tok_c = {c: 0 for c, _ in CLASSES}
    sx_tok_c = {c: np.zeros(D, dtype=np.float64) for c, _ in CLASSES}
    sxx_tok_c = {c: np.zeros((D, D), dtype=np.float64) for c, _ in CLASSES}

    # also pool per-window via attention pooling using REVE's cls_query_token
    print("Loading model for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(-1)
    scale = D ** -0.5
    del model

    n_win_all = 0
    sx_win = np.zeros(D, dtype=np.float64)
    sxx_win = np.zeros((D, D), dtype=np.float64)
    n_win_c = {c: 0 for c, _ in CLASSES}
    sx_win_c = {c: np.zeros(D, dtype=np.float64) for c, _ in CLASSES}
    sxx_win_c = {c: np.zeros((D, D), dtype=np.float64) for c, _ in CLASSES}

    for i, p in enumerate(files, 1):
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)         # (N,C,P,D)
            lbl = f["trial/label"][:]
        N, C, P, _ = lat.shape

        # per-token
        toks = lat.reshape(-1, D).astype(np.float64)
        n_tok_all += toks.shape[0]
        sx_tok += toks.sum(axis=0)
        sxx_tok += toks.T @ toks
        # per-token per-class (broadcast labels: each window has C*P tokens of same label)
        tok_lbl = np.repeat(lbl, C * P)
        for c, _ in CLASSES:
            m = tok_lbl == c
            if m.any():
                sub = toks[m]
                n_tok_c[c] += sub.shape[0]
                sx_tok_c[c] += sub.sum(axis=0)
                sxx_tok_c[c] += sub.T @ sub

        # per-window: attention pool with REVE's cls_query_token
        flat = lat.reshape(N, C * P, D)
        scores = (flat @ q) * scale
        w = softmax(scores, axis=1)
        emb = (w[..., None] * flat).sum(axis=1).astype(np.float64)   # (N, D)
        n_win_all += emb.shape[0]
        sx_win += emb.sum(axis=0)
        sxx_win += emb.T @ emb
        for c, _ in CLASSES:
            m = lbl == c
            if m.any():
                sub = emb[m]
                n_win_c[c] += sub.shape[0]
                sx_win_c[c] += sub.sum(axis=0)
                sxx_win_c[c] += sub.T @ sub

        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  tokens so far: {n_tok_all:,}", flush=True)

    cov_tok = cov_from_sums(n_tok_all, sx_tok, sxx_tok)
    eig_tok = np.linalg.eigvalsh(cov_tok)[::-1]

    cov_win = cov_from_sums(n_win_all, sx_win, sxx_win)
    eig_win = np.linalg.eigvalsh(cov_win)[::-1]

    # ---------- summary ----------
    lines = []
    lines.append("Rank/spectrum analysis on PAPER-preprocessed REVE latents")
    lines.append(f"latents from: {LAT_DIR}")
    lines.append("")

    lines.append("PER-TOKEN (each (channel, patch) is one 512-D sample):")
    lines.append(f"  N tokens = {n_tok_all:,}")
    lines.append(f"  eff_rank = {effective_rank(eig_tok):.2f}")
    lines.append(f"  k50={k_at(eig_tok,0.5)}  k90={k_at(eig_tok,0.9)}  k99={k_at(eig_tok,0.99)}")
    lines.append(f"  log10 top-10 eigvals: {np.log10(np.maximum(eig_tok[:10], 1e-12)).round(2).tolist()}")
    lines.append("")
    lines.append("  per-class:")
    eig_tok_c = {}
    for c, name in CLASSES:
        n = n_tok_c[c]
        if n < 100: continue
        cov_c = cov_from_sums(n, sx_tok_c[c], sxx_tok_c[c])
        eig_c = np.linalg.eigvalsh(cov_c)[::-1]
        eig_tok_c[c] = eig_c
        lines.append(f"    {name}: N={n}  eff_rank={effective_rank(eig_c):.2f}  "
                     f"k90={k_at(eig_c,0.9)}  k99={k_at(eig_c,0.99)}")
    lines.append("")

    lines.append("PER-WINDOW (attention-pooled to one 512-D vector per trial):")
    lines.append(f"  N windows = {n_win_all:,}")
    lines.append(f"  eff_rank = {effective_rank(eig_win):.2f}")
    lines.append(f"  k50={k_at(eig_win,0.5)}  k90={k_at(eig_win,0.9)}  k99={k_at(eig_win,0.99)}")
    lines.append(f"  log10 top-10 eigvals: {np.log10(np.maximum(eig_win[:10], 1e-12)).round(2).tolist()}")
    lines.append("")
    lines.append("  per-class:")
    eig_win_c = {}
    for c, name in CLASSES:
        n = n_win_c[c]
        if n < 50: continue
        cov_c = cov_from_sums(n, sx_win_c[c], sxx_win_c[c])
        eig_c = np.linalg.eigvalsh(cov_c)[::-1]
        eig_win_c[c] = eig_c
        lines.append(f"    {name}: N={n}  eff_rank={effective_rank(eig_c):.2f}  "
                     f"k90={k_at(eig_c,0.9)}  k99={k_at(eig_c,0.99)}")
    lines.append("")

    # active coords on top-5 PCs of grand per-token covariance
    w_, V_ = np.linalg.eigh(cov_tok)
    order = np.argsort(w_)[::-1]
    V5 = V_[:, order[:5]]
    union_thr = set()
    for thr in [0.10, 0.15, 0.20]:
        u = set()
        for k in range(5):
            u |= set(np.where(np.abs(V5[:, k]) > thr)[0].tolist())
        if thr == 0.10:
            union_thr = u
        lines.append(f"  active coords union over PCs 1..5 (|V|>{thr}): "
                     f"{len(u)} -> {sorted(u)[:15]}{'...' if len(u)>15 else ''}")
    lines.append("")
    lines.append(f"  participation ratios for top-8 PCs:")
    for k in range(8):
        v = V_[:, order[k]]
        lines.append(f"    PC{k+1}: {participation_ratio(v):6.2f} eff coords  "
                     f"eigval={w_[order[k]]:.3g}")
    lines.append("")

    msg = "\n".join(lines)
    OUT_TXT.write_text(msg + "\n")
    print()
    print(msg)
    print(f"\nwrote {OUT_TXT}")

    # ---------- PDF: heatmaps + spectra ----------
    with PdfPages(OUT_PDF) as pdf:
        # page 1: per-token grand
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        vmax = float(np.percentile(np.abs(cov_tok), 99))
        im = axes[0].imshow(cov_tok, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                            interpolation="nearest")
        axes[0].set_title(f"per-token cov (N={n_tok_all:,})\nclip ±{vmax:.2g}")
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        sd = np.sqrt(np.maximum(np.diag(cov_tok), 1e-30))
        corr = cov_tok / np.outer(sd, sd)
        im = axes[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        axes[1].set_title("per-token correlation")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        eff = effective_rank(eig_tok)
        axes[2].semilogy(np.maximum(eig_tok, 1e-12))
        axes[2].set_title(f"per-token spectrum  eff_rank={eff:.2f}  "
                          f"k90={k_at(eig_tok,0.9)}  k99={k_at(eig_tok,0.99)}")
        axes[2].set_xlabel("rank"); axes[2].set_ylabel("eigval (log)")
        axes[2].grid(True, which="both", alpha=0.3)

        fig.suptitle("PAPER-preprocessed REVE: per-token covariance summary", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # page 2: per-window attention-pooled grand
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        vmax = float(np.percentile(np.abs(cov_win), 99))
        im = axes[0].imshow(cov_win, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                            interpolation="nearest")
        axes[0].set_title(f"per-window cov (N={n_win_all:,})\nclip ±{vmax:.2g}")
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        sd = np.sqrt(np.maximum(np.diag(cov_win), 1e-30))
        corr_w = cov_win / np.outer(sd, sd)
        im = axes[1].imshow(corr_w, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        axes[1].set_title("per-window correlation")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        eff = effective_rank(eig_win)
        axes[2].semilogy(np.maximum(eig_win, 1e-12))
        axes[2].set_title(f"per-window spectrum  eff_rank={eff:.2f}  "
                          f"k90={k_at(eig_win,0.9)}  k99={k_at(eig_win,0.99)}")
        axes[2].set_xlabel("rank"); axes[2].set_ylabel("eigval (log)")
        axes[2].grid(True, which="both", alpha=0.3)

        fig.suptitle("PAPER-preprocessed REVE: per-window (attention-pooled) covariance",
                     fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # page 3: per-class spectra overlay (per-token and per-window)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        colors = ["C0", "C1", "C2", "C3"]
        for (c, name), col in zip(CLASSES, colors):
            if c in eig_tok_c:
                axes[0].semilogy(np.maximum(eig_tok_c[c], 1e-12), color=col, label=name, alpha=0.85)
            if c in eig_win_c:
                axes[1].semilogy(np.maximum(eig_win_c[c], 1e-12), color=col, label=name, alpha=0.85)
        axes[0].semilogy(np.maximum(eig_tok, 1e-12), color="k", linestyle="--", label="all", alpha=0.7)
        axes[1].semilogy(np.maximum(eig_win, 1e-12), color="k", linestyle="--", label="all", alpha=0.7)
        axes[0].set_title("per-token spectra"); axes[0].grid(True, which="both", alpha=0.3); axes[0].legend()
        axes[1].set_title("per-window spectra"); axes[1].grid(True, which="both", alpha=0.3); axes[1].legend()
        for a in axes:
            a.set_xlabel("rank"); a.set_ylabel("eigval (log)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # page 4: top-5 eigenvectors (signed)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        vmax = float(np.percentile(np.abs(V5), 99.5))
        im = axes[0].imshow(V5, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                            interpolation="nearest", aspect="auto")
        axes[0].set_title(f"top-5 eigvecs of per-token cov (V_K, signed, clip ±{vmax:.3f})")
        axes[0].set_xlabel("PC (1..5)"); axes[0].set_ylabel("embedding coord 0..511")
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        row_max = np.abs(V5).max(axis=1)
        axes[1].bar(np.arange(D), row_max, color="steelblue", width=1)
        axes[1].set_xlabel("embedding coord")
        axes[1].set_ylabel("max |V_K| over PCs 1..5")
        axes[1].set_title(f"per-coord max loading among top-5 PCs "
                          f"({(row_max>0.1).sum()} coords > 0.1)")
        for idx in np.argsort(row_max)[::-1][:8]:
            axes[1].text(idx, row_max[idx] + 0.01, str(idx), ha="center", fontsize=7, rotation=90)
        fig.suptitle("PAPER-preprocessed REVE: top-5 eigenvectors of per-token cov", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
