"""Visualize the top-K eigenvectors of REVE per-subject grand token covariance.

If REVE has axis-aligned outlier features (specific coordinates of the 512-D output
always carry the dominant variance), then |V_K| should look sparse: each of the top-K
columns concentrates on a specific embedding coordinate (row).
"""
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LAT_DIR = Path("/media/alex/DATA1/REVE/latents")
OUT_PDF = Path("/media/alex/DATA1/REVE/reports/reve_eigvec_heatmap.pdf")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_active_coords.npz")

D = 512
K = 50
THRESHOLD = 0.1   # for counting "active" coordinates


def grand_cov_eigvecs(path: Path, k: int):
    with h5py.File(path, "r") as f:
        lat = f["windows/trial/latent"][:].astype(np.float32)
        lbl = f["windows/trial/label"][:]
    m = lbl >= 0
    toks = lat[m].reshape(-1, D).astype(np.float64)
    Xc = toks - toks.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(toks.shape[0] - 1, 1)
    cov = (cov + cov.T) * 0.5
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx].astype(np.float32), w[idx].astype(np.float64)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    sids = [int(Path(p).stem[1:]) for p in files]
    sid_to_path = {int(Path(p).stem[1:]): p for p in files}

    target_sid = 1
    V_K, w_K = grand_cov_eigvecs(Path(sid_to_path[target_sid]), K)
    row_maxabs = np.abs(V_K).max(axis=1)
    active_mask = row_maxabs > THRESHOLD
    n_active = int(active_mask.sum())
    active_coords = np.where(active_mask)[0]

    sample_sids = [1, 22, 45, 67, 90, 109]
    sample_data = {}
    for s in sample_sids:
        if s in sid_to_path:
            V, w = grand_cov_eigvecs(Path(sid_to_path[s]), K)
            sample_data[s] = (V, w)

    # Cross-subject Jaccard on the active-coord sets
    active_sets = {}
    for s, (V, _) in sample_data.items():
        rm = np.abs(V).max(axis=1)
        active_sets[s] = set(np.where(rm > THRESHOLD)[0].tolist())

    with PdfPages(OUT_PDF) as pdf:
        # Page 1: signed V_K, |V_K|, eigenvalues, per-coord max loading
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.7, 0.7], wspace=0.35,
                              left=0.05, right=0.97, top=0.92, bottom=0.08)

        ax = fig.add_subplot(gs[0, 0])
        vmax = float(np.percentile(np.abs(V_K), 99.5))
        im = ax.imshow(V_K, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_title(f"V_K signed  (clip ±{vmax:.3f})", fontsize=10)
        ax.set_xlabel("principal component (1..K)")
        ax.set_ylabel("embedding coord (0..511)")
        plt.colorbar(im, ax=ax, fraction=0.04)

        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(np.abs(V_K), cmap="viridis", vmin=0, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_title("|V_K|", fontsize=10)
        ax.set_xlabel("principal component")
        plt.colorbar(im, ax=ax, fraction=0.04)

        ax = fig.add_subplot(gs[0, 2])
        ax.semilogy(np.arange(1, K + 1), w_K, "-o", markersize=4, color="C0")
        ax.set_xlabel("rank"); ax.set_ylabel("eigenvalue (log)")
        ax.set_title(f"top-{K} eigenvalues", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

        ax = fig.add_subplot(gs[0, 3])
        ax.barh(np.arange(D), row_maxabs, color="steelblue", height=1.0)
        ax.invert_yaxis()
        ax.set_xlabel("max |V_K| over PCs")
        ax.set_title(f"per-coord max loading\n{n_active} coords > {THRESHOLD}", fontsize=10)

        fig.suptitle(
            f"S{target_sid:03d} — top-{K} eigenvectors of REVE grand token covariance",
            fontsize=12,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: |V_K| side-by-side for several subjects (axis-alignment invariance)
        n = len(sample_data)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 7), sharey=True)
        for ax, (sid, (V, _)) in zip(axes, sample_data.items()):
            vmax = float(np.percentile(np.abs(V), 99.5))
            ax.imshow(np.abs(V), cmap="viridis", vmin=0, vmax=vmax,
                      interpolation="nearest", aspect="auto")
            ax.set_title(f"S{sid:03d}\n|V_K|", fontsize=10)
            ax.set_xlabel("PC")
        axes[0].set_ylabel("embedding coord")
        fig.suptitle(
            f"|V_K| (top-{K} eigvecs of grand cov) — same active coordinates across subjects?",
            fontsize=11,
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: per-coord max loading across all 512 coords for target subject
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(np.arange(D), row_maxabs, color="steelblue", width=1.0)
        ax.set_xlabel("embedding coordinate (0..511)")
        ax.set_ylabel(f"max |V_K| over top-{K} PCs")
        ax.set_title(
            f"S{target_sid:03d}: per-coord max loading among top-{K} eigenvectors  "
            f"({n_active} coords > {THRESHOLD})",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        top10 = np.argsort(row_maxabs)[::-1][:10]
        for i in top10:
            ax.text(i, row_maxabs[i] + 0.015, str(i),
                    ha="center", fontsize=7, rotation=90)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: active-coord overlap across sample subjects
        fig, ax = plt.subplots(figsize=(8, 6))
        sids_list = list(active_sets.keys())
        m = len(sids_list)
        J = np.zeros((m, m))
        for i, a in enumerate(sids_list):
            for j, b in enumerate(sids_list):
                A = active_sets[a]; B = active_sets[b]
                u = len(A | B); J[i, j] = (len(A & B) / u) if u else 1.0
        im = ax.imshow(J, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(m)); ax.set_yticks(range(m))
        ax.set_xticklabels([f"S{s:03d}" for s in sids_list], rotation=45)
        ax.set_yticklabels([f"S{s:03d}" for s in sids_list])
        for i in range(m):
            for j in range(m):
                ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center",
                        color="white" if J[i, j] < 0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title(
            f"Jaccard overlap of active-coord sets (|V_K|>{THRESHOLD}) across subjects",
            fontsize=11,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    np.savez_compressed(
        OUT_NPZ,
        active_coords=active_coords,
        row_maxabs_S001=row_maxabs.astype(np.float32),
        threshold=np.float64(THRESHOLD),
        K=np.int64(K),
        sample_sids=np.array(list(active_sets.keys()), dtype=np.int64),
    )

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_NPZ}")
    print()
    print(f"S{target_sid:03d}: {n_active}/{D} coordinates with max |V_K| > {THRESHOLD}")
    print(f"top-15 coords (sorted by max loading): "
          f"{sorted(np.argsort(row_maxabs)[::-1][:15].tolist())}")
    print()
    print(f"top-{K} eigenvalues (log10): "
          f"{np.log10(np.maximum(w_K, 1e-12)).round(2).tolist()}")
    print()
    print("Cross-subject Jaccard of active-coord sets among sampled subjects:")
    sids_list = list(active_sets.keys())
    for i, a in enumerate(sids_list):
        row = []
        for b in sids_list:
            u = len(active_sets[a] | active_sets[b])
            row.append(f"{(len(active_sets[a] & active_sets[b]) / u if u else 1):.2f}")
        print(f"  S{a:03d}: " + "  ".join(row))


if __name__ == "__main__":
    main()
