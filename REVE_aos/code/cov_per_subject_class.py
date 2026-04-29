"""Per-subject per-class REVE token covariance.

Outputs a single PDF with:
- N pages of (5 subjects per page) x (5 columns):
    cols 0-3 = covariance heatmaps for L, R, B, F
    col 4    = bar chart of effective rank per class for that subject
- 1 final page with 4 histograms of effective rank per class across all subjects
  (each histogram's bar heights sum to n_subjects).

Saves the (n_subjects, 4) effective-rank matrix to NPZ for downstream use.
"""
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LAT_DIR = Path("/media/alex/DATA1/REVE/latents")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_per_subject_class_cov.pdf")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_per_subject_class_eff_rank.npz")

CLASSES = [(0, "L"), (1, "R"), (2, "B"), (3, "F")]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
D = 512
SUBJECTS_PER_PAGE = 5


def effective_rank(eigvals: np.ndarray) -> float:
    e = np.maximum(eigvals, 0.0)
    s = e.sum()
    if s <= 0:
        return 0.0
    p = e / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def k_at(eigvals: np.ndarray, frac: float) -> int:
    e = np.maximum(eigvals, 0.0)
    s = e.sum()
    if s <= 0:
        return 0
    cum = np.cumsum(e) / s
    return int(np.searchsorted(cum, frac) + 1)


def per_subject_class_covs(path: Path):
    """Returns {class_label: (cov 512x512 fp32, eff_rank, k90, k99, n_tokens) | None}."""
    with h5py.File(path, "r") as f:
        lat = f["windows/trial/latent"][:].astype(np.float32)   # (N, C, P, D)
        lbl = f["windows/trial/label"][:]                        # (N,)
    out = {}
    for c, _ in CLASSES:
        mask = lbl == c
        if not mask.any():
            out[c] = None
            continue
        toks = lat[mask].reshape(-1, D).astype(np.float64)
        n = toks.shape[0]
        toks -= toks.mean(axis=0, keepdims=True)
        cov = (toks.T @ toks) / max(n - 1, 1)
        cov = (cov + cov.T) * 0.5
        eig = np.linalg.eigvalsh(cov)[::-1]
        out[c] = (
            cov.astype(np.float32),
            effective_rank(eig),
            k_at(eig, 0.90),
            k_at(eig, 0.99),
            int(n),
        )
    return out


def render_subjects_page(pdf, subjects, results_by_subject, page_idx, n_pages):
    n_rows = len(subjects)
    fig = plt.figure(figsize=(14, 2.6 * n_rows + 1.0))
    gs = fig.add_gridspec(
        n_rows, 5,
        width_ratios=[1, 1, 1, 1, 1.15],
        left=0.06, right=0.99, top=0.93, bottom=0.04,
        hspace=0.5, wspace=0.20,
    )
    fig.suptitle(
        f"REVE token covariance per subject × class — page {page_idx}/{n_pages}",
        fontsize=11,
    )

    for row, sid in enumerate(subjects):
        res = results_by_subject[sid]

        # one shared color scale per row, so cross-class contrasts within a subject are honest
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

        # rank stats column: bar chart of effective rank per class
        ax_st = fig.add_subplot(gs[row, 4])
        effs = []
        labels = []
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


def render_histograms(pdf, eff_ranks: np.ndarray):
    """eff_ranks: (n_subjects, 4). One histogram per class, bars sum to n_subjects."""
    n_subj = eff_ranks.shape[0]
    finite = eff_ranks[np.isfinite(eff_ranks)]
    lo = float(np.floor(finite.min() * 10) / 10)
    hi = float(np.ceil(finite.max() * 10) / 10)
    bins = np.linspace(lo, hi, 26)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    fig.suptitle(
        f"Effective rank of REVE token covariance per class — distribution across {n_subj} subjects",
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
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    sids = [int(Path(p).stem[1:]) for p in files]
    n = len(files)
    print(f"{n} subjects, {SUBJECTS_PER_PAGE} per page")

    eff_ranks = np.full((n, 4), np.nan, dtype=np.float32)
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}

    n_subj_pages = (n + SUBJECTS_PER_PAGE - 1) // SUBJECTS_PER_PAGE
    n_total_pages = n_subj_pages + 1

    with PdfPages(OUT_PDF) as pdf:
        for page in range(n_subj_pages):
            i0 = page * SUBJECTS_PER_PAGE
            i1 = min(i0 + SUBJECTS_PER_PAGE, n)
            page_files = files[i0:i1]
            page_sids  = sids[i0:i1]
            results_by_subject = {}
            for path, sid in zip(page_files, page_sids):
                res = per_subject_class_covs(Path(path))
                results_by_subject[sid] = res
                for col, (c, _) in enumerate(CLASSES):
                    if res[c] is not None:
                        eff_ranks[sid_to_idx[sid], col] = res[c][1]
            render_subjects_page(pdf, page_sids, results_by_subject, page + 1, n_total_pages)
            print(f"  page {page+1}/{n_total_pages}: S{page_sids[0]:03d}–S{page_sids[-1]:03d}")
            del results_by_subject

        render_histograms(pdf, eff_ranks)
        print(f"  page {n_total_pages}/{n_total_pages}: histograms")

    np.savez_compressed(OUT_NPZ, eff_ranks=eff_ranks, subject_ids=np.array(sids))
    print(f"\nwrote {OUT_PDF}")
    print(f"wrote {OUT_NPZ}")
    print(f"\nper-class effective rank summary (across {n} subjects):")
    for col, (_, name) in enumerate(CLASSES):
        e = eff_ranks[:, col]
        e = e[np.isfinite(e)]
        print(f"  {name}: N={len(e)}  median={np.median(e):.2f}  "
              f"mean={e.mean():.2f}  min={e.min():.2f}  max={e.max():.2f}")


if __name__ == "__main__":
    main()
