"""Per-subject per-class REVE token covariance AFTER removing the top-K shared directions.

The top-K principal directions of the grand covariance dominate the variance and look
nearly identical across all subjects and classes. We project those directions out and
recompute the per-subject per-class covariances on the residual subspace — that's where
any class-discriminative structure must live.

Reads the grand covariance computed by cov_glance.py.
Outputs:
  reve_per_subject_class_cov_residual_K{K}.pdf
  reve_per_subject_class_eff_rank_residual_K{K}.npz
  reve_residual_projector_K{K}.npz   (V_K and the projector for downstream use)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LAT_DIR = Path("/media/alex/DATA1/REVE/latents")
GRAND   = Path("/media/alex/DATA1/REVE/reports/reve_token_covariance.npz")
OUT_DIR = Path("/media/alex/DATA1/REVE/reports")

CLASSES = [(0, "L"), (1, "R"), (2, "B"), (3, "F")]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
D = 512
SUBJECTS_PER_PAGE = 5


def effective_rank(eigvals):
    e = np.maximum(eigvals, 0.0); s = e.sum()
    if s <= 0: return 0.0
    p = e / s; p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def k_at(eigvals, frac):
    e = np.maximum(eigvals, 0.0); s = e.sum()
    if s <= 0: return 0
    cum = np.cumsum(e) / s
    return int(np.searchsorted(cum, frac) + 1)


def build_projector(K: int):
    """Eigendecompose grand cov from cov_glance.npz, return (V_K, P_resid, eig_all)."""
    npz = np.load(GRAND)
    cov_all = npz["cov_all"]
    w, V = np.linalg.eigh(cov_all)        # ascending
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    V_K = V[:, :K]                         # top-K eigenvectors
    P_resid = np.eye(D) - V_K @ V_K.T      # residual projector (D, D), symmetric
    return V_K, P_resid, w


def per_subject_class_covs_residual(path: Path, P_resid: np.ndarray):
    with h5py.File(path, "r") as f:
        lat = f["windows/trial/latent"][:].astype(np.float32)  # (N, C, P, D)
        lbl = f["windows/trial/label"][:]
    out = {}
    for c, _ in CLASSES:
        mask = lbl == c
        if not mask.any():
            out[c] = None; continue
        toks = lat[mask].reshape(-1, D).astype(np.float64)
        toks = toks @ P_resid                          # project to residual subspace
        n = toks.shape[0]
        toks -= toks.mean(axis=0, keepdims=True)
        cov = (toks.T @ toks) / max(n - 1, 1)
        cov = (cov + cov.T) * 0.5
        eig = np.linalg.eigvalsh(cov)[::-1]
        out[c] = (cov.astype(np.float32),
                  effective_rank(eig),
                  k_at(eig, 0.90),
                  k_at(eig, 0.99),
                  int(n))
    return out


def render_subjects_page(pdf, subjects, results_by_subject, page_idx, n_pages, K):
    n_rows = len(subjects)
    fig = plt.figure(figsize=(14, 2.6 * n_rows + 1.0))
    gs = fig.add_gridspec(n_rows, 5, width_ratios=[1, 1, 1, 1, 1.15],
                          left=0.06, right=0.99, top=0.93, bottom=0.04,
                          hspace=0.5, wspace=0.20)
    fig.suptitle(
        f"REVE token covariance per subject × class — top-{K} dirs removed — "
        f"page {page_idx}/{n_pages}",
        fontsize=11,
    )
    for row, sid in enumerate(subjects):
        res = results_by_subject[sid]
        abs_vals = []
        for c, _ in CLASSES:
            if res[c] is not None:
                abs_vals.append(np.abs(res[c][0]).ravel())
        vmax = float(np.percentile(np.concatenate(abs_vals), 99)) if abs_vals else 1.0

        for col, (c, name) in enumerate(CLASSES):
            ax = fig.add_subplot(gs[row, col])
            r = res[c]
            if r is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.imshow(r[0], cmap="coolwarm", vmin=-vmax, vmax=vmax,
                          interpolation="nearest", aspect="equal", rasterized=True)
                ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"class {name}", fontsize=10)
            if col == 0:
                ax.text(-0.18, 0.5, f"S{sid:03d}", transform=ax.transAxes,
                        fontsize=10, ha="right", va="center")

        ax_st = fig.add_subplot(gs[row, 4])
        effs, labels = [], []
        for c, name in CLASSES:
            r = res[c]
            effs.append(r[1] if r is not None else 0.0)
            labels.append(name)
        bars = ax_st.bar(labels, effs, color=CLASS_COLORS, edgecolor="black", linewidth=0.5)
        ax_st.set_ylabel("eff. rank", fontsize=8)
        ax_st.tick_params(axis="both", labelsize=8)
        ax_st.set_ylim(0, max(effs) * 1.30 if max(effs) > 0 else 1)
        for bar, e in zip(bars, effs):
            ax_st.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f"{e:.1f}", ha="center", va="bottom", fontsize=7)
        ax_st.set_title(f"vmax={vmax:.2g}", fontsize=8)

    pdf.savefig(fig, bbox_inches="tight", dpi=110)
    plt.close(fig)


def render_histograms(pdf, eff_ranks: np.ndarray, K: int):
    n_subj = eff_ranks.shape[0]
    finite = eff_ranks[np.isfinite(eff_ranks)]
    lo = float(np.floor(finite.min() * 10) / 10)
    hi = float(np.ceil(finite.max() * 10) / 10)
    bins = np.linspace(lo, hi, 30)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    fig.suptitle(
        f"Effective rank of REVE token covariance (top-{K} dirs removed) — "
        f"distribution across {n_subj} subjects",
        fontsize=11,
    )
    for col, (_, name) in enumerate(CLASSES):
        ax = axes[col]
        e = eff_ranks[:, col]
        e = e[np.isfinite(e)]
        ax.hist(e, bins=bins, color=CLASS_COLORS[col], edgecolor="black", alpha=0.85)
        ax.set_title(f"class {name}\nN={len(e)}  median={np.median(e):.2f}  mean={e.mean():.2f}")
        ax.set_xlabel("effective rank")
        if col == 0:
            ax.set_ylabel("# subjects")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5,
                    help="number of dominant directions to remove (default 5)")
    args = ap.parse_args()
    K = args.K

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / f"reve_per_subject_class_cov_residual_K{K}.pdf"
    out_npz = OUT_DIR / f"reve_per_subject_class_eff_rank_residual_K{K}.npz"
    out_proj = OUT_DIR / f"reve_residual_projector_K{K}.npz"

    print(f"Building residual projector with K={K} ...")
    V_K, P_resid, eig_all = build_projector(K)
    var_kept = 1.0 - eig_all[:K].sum() / eig_all.sum()
    print(f"  kept variance fraction (after removing top-{K}): {var_kept*100:.4f}%")
    np.savez_compressed(out_proj, V_K=V_K, P_resid=P_resid,
                        eig_all=eig_all, K=np.int64(K))
    print(f"  saved projector to {out_proj}")

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    sids = [int(Path(p).stem[1:]) for p in files]
    n = len(files)
    print(f"{n} subjects, {SUBJECTS_PER_PAGE} per page")

    eff_ranks = np.full((n, 4), np.nan, dtype=np.float32)
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}

    n_subj_pages = (n + SUBJECTS_PER_PAGE - 1) // SUBJECTS_PER_PAGE
    n_total_pages = n_subj_pages + 1

    with PdfPages(out_pdf) as pdf:
        for page in range(n_subj_pages):
            i0 = page * SUBJECTS_PER_PAGE
            i1 = min(i0 + SUBJECTS_PER_PAGE, n)
            page_files = files[i0:i1]
            page_sids  = sids[i0:i1]
            results_by_subject = {}
            for path, sid in zip(page_files, page_sids):
                res = per_subject_class_covs_residual(Path(path), P_resid)
                results_by_subject[sid] = res
                for col, (c, _) in enumerate(CLASSES):
                    if res[c] is not None:
                        eff_ranks[sid_to_idx[sid], col] = res[c][1]
            render_subjects_page(pdf, page_sids, results_by_subject,
                                 page + 1, n_total_pages, K)
            print(f"  page {page+1}/{n_total_pages}: S{page_sids[0]:03d}-S{page_sids[-1]:03d}")
            del results_by_subject

        render_histograms(pdf, eff_ranks, K)
        print(f"  page {n_total_pages}/{n_total_pages}: histograms")

    np.savez_compressed(out_npz, eff_ranks=eff_ranks, subject_ids=np.array(sids),
                        K=np.int64(K))
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_npz}")
    print(f"\nper-class effective rank summary (residual, K={K}):")
    for col, (_, name) in enumerate(CLASSES):
        e = eff_ranks[:, col]; e = e[np.isfinite(e)]
        print(f"  {name}: N={len(e)}  median={np.median(e):.2f}  "
              f"mean={e.mean():.2f}  min={e.min():.2f}  max={e.max():.2f}  "
              f"std={e.std():.2f}")


if __name__ == "__main__":
    main()
