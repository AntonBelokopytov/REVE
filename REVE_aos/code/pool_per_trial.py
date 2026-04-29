"""Apply REVE's attention_pooling head to the saved per-token latents.

The encoder's forward() returns per-token tensors of shape (B, C, P, 512). REVE's
classification probe uses a separate attention_pooling step that compresses this to
one (B, 512) vector per window using a learned cls_query_token.

We compute that pooling offline from the saved tokens (no need to re-run the encoder).
Then we re-do the rank / spectrum analysis on the *per-trial* 512-D vectors.

Outputs:
  reve_pertrial_embeddings.npz       (embeddings, labels, subjects)
  reve_pertrial_rank_summary.txt     (rank / spectrum stats)
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
import numpy as np
import torch
from transformers import AutoModel

LAT_DIR = Path("/media/alex/DATA1/REVE/latents")
OUT_NPZ = Path("/media/alex/DATA1/REVE/reports/reve_pertrial_embeddings.npz")
OUT_TXT = Path("/media/alex/DATA1/REVE/reports/reve_pertrial_rank_summary.txt")

D = 512


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


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


def main():
    print("Loading REVE to extract cls_query_token ...")
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(D)        # (512,)
    scale = D ** -0.5
    print(f"  cls_query_token shape: {q.shape}, norm={np.linalg.norm(q):.3f}")

    files = sorted(glob.glob(str(LAT_DIR / "S*.h5")))
    print(f"{len(files)} subjects")

    all_emb, all_lbl, all_sub = [], [], []
    for i, path in enumerate(files, 1):
        sid = int(Path(path).stem[1:])
        with h5py.File(path, "r") as f:
            tokens = f["windows/trial/latent"][:].astype(np.float32)  # (N, C, P, E)
            labels = f["windows/trial/label"][:]
        N, C, P, E = tokens.shape
        flat = tokens.reshape(N, C * P, E)                            # (N, 256, 512)
        scores = (flat @ q) * scale                                   # (N, 256)
        weights = softmax(scores, axis=1)                             # (N, 256)
        emb = (weights[..., None] * flat).sum(axis=1)                 # (N, 512)

        all_emb.append(emb.astype(np.float32))
        all_lbl.append(labels)
        all_sub.append(np.full(N, sid, dtype=np.int32))
        if i % 20 == 0 or i == len(files):
            print(f"  pooled {i}/{len(files)}")

    emb = np.concatenate(all_emb)        # (Ntot, 512)
    lbl = np.concatenate(all_lbl)
    sub = np.concatenate(all_sub)
    print(f"\ntotal trials: {emb.shape[0]}, embedding dim: {emb.shape[1]}")
    print(f"label counts: {dict(zip(*np.unique(lbl, return_counts=True)))}")

    np.savez_compressed(OUT_NPZ, embeddings=emb, labels=lbl, subjects=sub)
    print(f"wrote {OUT_NPZ}")

    # rank summary, all trials
    Xc = emb.astype(np.float64) - emb.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (emb.shape[0] - 1)
    cov = (cov + cov.T) * 0.5
    eig = np.linalg.eigvalsh(cov)[::-1]
    eff = effective_rank(eig)
    k50, k90, k99 = k_at(eig, 0.5), k_at(eig, 0.9), k_at(eig, 0.99)

    lines = []
    lines.append("=== Per-trial REVE embedding rank summary ===")
    lines.append(f"N trials = {emb.shape[0]}, dim = 512")
    lines.append("")
    lines.append("Pooled (all classes incl rest):")
    lines.append(f"  effective rank = {eff:.2f}")
    lines.append(f"  k50 = {k50}   k90 = {k90}   k99 = {k99}")
    lines.append(f"  top-10 eigvals: {eig[:10].round(2).tolist()}")
    lines.append(f"  log10 top-10:   {np.log10(np.maximum(eig[:10], 1e-12)).round(2).tolist()}")
    lines.append("")

    # per-class
    for c, name in [(-1, "rest"), (0, "L"), (1, "R"), (2, "B"), (3, "F")]:
        m = lbl == c
        if m.sum() < 50:
            continue
        E = emb[m].astype(np.float64)
        E -= E.mean(axis=0, keepdims=True)
        cov_c = (E.T @ E) / (m.sum() - 1)
        cov_c = (cov_c + cov_c.T) * 0.5
        eig_c = np.linalg.eigvalsh(cov_c)[::-1]
        lines.append(f"class {name} (N={m.sum()}):")
        lines.append(f"  eff_rank={effective_rank(eig_c):.2f}  "
                     f"k50={k_at(eig_c, 0.5)}  k90={k_at(eig_c, 0.9)}  k99={k_at(eig_c, 0.99)}")

    out = "\n".join(lines)
    OUT_TXT.write_text(out + "\n")
    print()
    print(out)
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
