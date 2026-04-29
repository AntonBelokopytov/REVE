"""ICA component spectrum after projecting REVE latents to the top-100 V/U subspaces.

This is the ICA analogue of spectrum_full_512.py, but avoids 512-D ICA:

1. Load the same paper latents used by Experiment 7.
2. Load V_full and U_full from the Experiment 7 eigenspace file.
3. For each subject, immediately project:
     tokens (N, 256, 512) -> tokens_V100 (N, 256, 100)
     CLS    (N, 512)      -> cls_U100    (N, 100)
   so raw 512-D arrays are not accumulated in memory.
4. Fit FastICA on projected training-subject data only (subjects 1..70).
5. Transform all subjects into ICA coordinates.
6. For each ICA component k in 0..99, train the same per-component probes as
   Experiment 7:
     feature_k = [cls_ica_k, token_ica_k for each of 256 tokens]  # 257-D

Task split matches the paper split used by train_paper_head.py and
spectrum_full_512.py:
  train subjects 1..70, validation 71..89, test 90..109.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/media/alex/DATA/REVE/hf_cache")
for _TOK in (
    Path("/media/alex/DATA1/REVE/.hf_token"),
    Path("/media/alex/DATA/REVE/.hf_token"),
    Path("/media/alex/DATA/REVE/HugginFace_token.txt"),
):
    if _TOK.exists():
        os.environ.setdefault("HF_TOKEN", _TOK.read_text().strip())
        break

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModel


D = 512
K_ICA = 100
N_TOK = 256
LAT_DIR = Path("/media/alex/DATA1/REVE/latents_paper")
REPORT_DIR = Path("/media/alex/DATA1/REVE/reports")
VU_NPZ_NAME = "reve_VU_full.npz"
ICA_NPZ_NAME = "reve_ica_V100_train_only.npz"
PROJ_NPZ_NAME = "reve_ica_V100_proj.npz"
OUT_NPZ_NAME = "reve_ica_V100_spectrum.npz"
OUT_PDF_NAME = "reve_ica_V100_spectrum.pdf"
OUT_TXT_NAME = "reve_ica_V100_spectrum.txt"
EPOCHS = 20
BATCH = 64
LR = 1e-4
WD = 1e-2
DROPOUT = 0.1
PATIENCE = 5
SEED = 0


def softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def load_cls_query() -> tuple[np.ndarray, float]:
    print("Loading model for cls_query_token ...", flush=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    q = model.cls_query_token.detach().cpu().numpy().reshape(-1).astype(np.float32)
    del model
    return q, D ** -0.5


def files_for(lat_dir: Path) -> list[Path]:
    files = [Path(p) for p in sorted(glob.glob(str(lat_dir / "S*.h5")))]
    if not files:
        raise FileNotFoundError(f"No S*.h5 files found in {lat_dir}")
    return files


def sid_for(path: Path) -> int:
    return int(path.stem[1:])


def load_subspaces(vu_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(vu_npz)
    V100 = d["V_full"][:, :K_ICA].astype(np.float32)
    U100 = d["U_full"][:, :K_ICA].astype(np.float32)
    print(f"Loaded V100/U100 from {vu_npz}: {V100.shape}, {U100.shape}", flush=True)
    return V100, U100


def collect_training_projected_samples(
    files: list[Path],
    q: np.ndarray,
    scale: float,
    V100: np.ndarray,
    U100: np.ndarray,
    max_token_fit_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit samples are training subjects only, already projected to 100-D."""
    rng = np.random.default_rng(SEED)
    train_sids = set(range(1, 71))
    train_files = [p for p in files if sid_for(p) in train_sids]
    token_blocks: list[np.ndarray] = []
    cls_blocks: list[np.ndarray] = []
    total_tokens = 0
    t0 = time.time()

    print(f"Collecting projected ICA fit samples from {len(train_files)} training subjects ...", flush=True)
    for i, p in enumerate(train_files, 1):
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)
        n = lat.shape[0]
        flat = lat.reshape(n, N_TOK, D)

        scores = (flat @ q) * scale
        w = softmax_np(scores, axis=1)
        cls = (w[..., None] * flat).sum(axis=1)

        tok_v100 = (flat.reshape(-1, D) @ V100).astype(np.float32)
        cls_u100 = (cls @ U100).astype(np.float32)
        total_tokens += tok_v100.shape[0]

        if max_token_fit_samples > 0 and tok_v100.shape[0] > max_token_fit_samples:
            take = rng.choice(tok_v100.shape[0], size=max_token_fit_samples, replace=False)
            tok_v100 = tok_v100[take]

        token_blocks.append(tok_v100)
        cls_blocks.append(cls_u100)

        if i % 10 == 0 or i == len(train_files):
            print(f"  fit samples {i}/{len(train_files)}  elapsed={time.time() - t0:.0f}s", flush=True)

    X_tok = np.concatenate(token_blocks, axis=0)
    Y_cls = np.concatenate(cls_blocks, axis=0)
    if max_token_fit_samples > 0 and X_tok.shape[0] > max_token_fit_samples:
        take = rng.choice(X_tok.shape[0], size=max_token_fit_samples, replace=False)
        X_tok = X_tok[take]

    print(f"  token V100 fit samples: {X_tok.shape[0]:,} of {total_tokens:,} training tokens", flush=True)
    print(f"  CLS U100 fit samples  : {Y_cls.shape[0]:,} training windows", flush=True)
    return X_tok, Y_cls


def fit_or_load_ica(
    ica_npz: Path,
    files: list[Path],
    q: np.ndarray,
    scale: float,
    V100: np.ndarray,
    U100: np.ndarray,
    max_token_fit_samples: int,
    max_iter: int,
    tol: float,
) -> tuple[FastICA, FastICA]:
    if ica_npz.exists():
        d = np.load(ica_npz)
        token_ica = FastICA(n_components=K_ICA, whiten="unit-variance", random_state=SEED)
        cls_ica = FastICA(n_components=K_ICA, whiten="unit-variance", random_state=SEED)
        token_ica.components_ = d["token_components"]
        token_ica.mixing_ = d["token_mixing"]
        token_ica.mean_ = d["token_mean"]
        token_ica.whitening_ = d["token_whitening"]
        token_ica.n_features_in_ = K_ICA
        token_ica.n_iter_ = int(d["token_n_iter"])
        cls_ica.components_ = d["cls_components"]
        cls_ica.mixing_ = d["cls_mixing"]
        cls_ica.mean_ = d["cls_mean"]
        cls_ica.whitening_ = d["cls_whitening"]
        cls_ica.n_features_in_ = K_ICA
        cls_ica.n_iter_ = int(d["cls_n_iter"])
        print(f"Loaded ICA transforms from {ica_npz}", flush=True)
        return token_ica, cls_ica

    X_tok, Y_cls = collect_training_projected_samples(
        files, q, scale, V100, U100, max_token_fit_samples
    )
    kwargs = dict(
        n_components=K_ICA,
        algorithm="parallel",
        whiten="unit-variance",
        whiten_solver="svd",
        random_state=SEED,
        max_iter=max_iter,
        tol=tol,
    )

    print("Fitting token FastICA on training V100 tokens only ...", flush=True)
    token_ica = FastICA(**kwargs).fit(X_tok)
    print(f"  token ICA iterations: {token_ica.n_iter_}", flush=True)

    print("Fitting CLS FastICA on training U100 CLS vectors only ...", flush=True)
    cls_ica = FastICA(**kwargs).fit(Y_cls)
    print(f"  CLS ICA iterations: {cls_ica.n_iter_}", flush=True)

    ica_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ica_npz,
        token_components=token_ica.components_.astype(np.float32),
        token_mixing=token_ica.mixing_.astype(np.float32),
        token_mean=token_ica.mean_.astype(np.float32),
        token_whitening=token_ica.whitening_.astype(np.float32),
        token_n_iter=np.int64(token_ica.n_iter_),
        cls_components=cls_ica.components_.astype(np.float32),
        cls_mixing=cls_ica.mixing_.astype(np.float32),
        cls_mean=cls_ica.mean_.astype(np.float32),
        cls_whitening=cls_ica.whitening_.astype(np.float32),
        cls_n_iter=np.int64(cls_ica.n_iter_),
        max_token_fit_samples=np.int64(max_token_fit_samples),
        seed=np.int64(SEED),
        K=np.int64(K_ICA),
    )
    print(f"Saved ICA transforms to {ica_npz}", flush=True)
    return token_ica, cls_ica


def transform_fastica(ica: FastICA, x: np.ndarray) -> np.ndarray:
    return (x - ica.mean_) @ ica.components_.T


def compute_or_load_ica_projections(
    proj_npz: Path,
    files: list[Path],
    q: np.ndarray,
    scale: float,
    V100: np.ndarray,
    U100: np.ndarray,
    token_ica: FastICA,
    cls_ica: FastICA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if proj_npz.exists():
        d = np.load(proj_npz)
        return d["ica_tokens"], d["ica_cls"], d["labels"], d["subjects"]

    n_total = 0
    for p in files:
        with h5py.File(p, "r") as f:
            n_total += f["trial/latent"].shape[0]

    print(f"Allocating ICA tokens fp16 ({n_total}, {N_TOK}, {K_ICA}) "
          f"= {n_total * N_TOK * K_ICA * 2 / 1e9:.2f} GB", flush=True)
    ica_tokens = np.empty((n_total, N_TOK, K_ICA), dtype=np.float16)
    ica_cls = np.empty((n_total, K_ICA), dtype=np.float32)
    labels = np.empty(n_total, dtype=np.int8)
    subjects = np.empty(n_total, dtype=np.int32)

    pos = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        sid = sid_for(p)
        with h5py.File(p, "r") as f:
            lat = f["trial/latent"][:].astype(np.float32)
            lab = f["trial/label"][:]
        n = lat.shape[0]
        flat = lat.reshape(n, N_TOK, D)

        scores = (flat @ q) * scale
        w = softmax_np(scores, axis=1)
        cls = (w[..., None] * flat).sum(axis=1)

        tok_v100 = flat.reshape(-1, D) @ V100
        cls_u100 = cls @ U100
        tok_ica = transform_fastica(token_ica, tok_v100).reshape(n, N_TOK, K_ICA)
        cls_ica_coords = transform_fastica(cls_ica, cls_u100)

        ica_tokens[pos:pos + n] = tok_ica.astype(np.float16)
        ica_cls[pos:pos + n] = cls_ica_coords.astype(np.float32)
        labels[pos:pos + n] = lab
        subjects[pos:pos + n] = sid
        pos += n

        if i % 20 == 0 or i == len(files):
            print(f"  projected {i}/{len(files)}  elapsed={time.time() - t0:.0f}s", flush=True)

    np.savez(
        proj_npz,
        ica_tokens=ica_tokens,
        ica_cls=ica_cls,
        labels=labels,
        subjects=subjects,
    )
    print(f"Saved ICA projections to {proj_npz}", flush=True)
    return ica_tokens, ica_cls, labels, subjects


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class Head(nn.Module):
    def __init__(self, n_features, n_classes, dropout=DROPOUT):
        super().__init__()
        self.norm = RMSNorm(n_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.fc(self.drop(self.norm(x)))


def iterate(X, y, batch=BATCH, shuffle=True, rng=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        sl = idx[i:i + batch]
        yield torch.from_numpy(X[sl].astype(np.float32)), torch.from_numpy(y[sl])


def train_probe(Xtr, ytr, Xva, yva, Xte, yte, n_classes):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Head(Xtr.shape[1], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_state, bad = -1.0, None, 0

    for _epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in iterate(Xtr, ytr, shuffle=True, rng=rng):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in iterate(Xva, yva, shuffle=False):
                preds.append(model(xb).argmax(-1).numpy())
        preds = np.concatenate(preds)
        val_bal = balanced_accuracy_score(yva, preds)
        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in iterate(Xte, yte, shuffle=False):
            preds.append(model(xb).argmax(-1).numpy())
    preds = np.concatenate(preds)
    return float(balanced_accuracy_score(yte, preds))


def plot_results(out_pdf: Path, task_bal, sub_bal, n_subjects):
    k = np.arange(K_ICA)
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    ax.plot(k, task_bal, "-o", color="C0", label="task (4-class)", linewidth=1, markersize=3)
    ax.plot(k, sub_bal, "-o", color="C3", label="subject (108-class)", linewidth=1, markersize=3)
    ax.axhline(0.25, color="C0", linestyle=":", alpha=0.5, label="task chance")
    ax.axhline(1 / n_subjects, color="C3", linestyle=":", alpha=0.5, label="subject chance")
    ax.set_xlabel("ICA component index k after V/U100 projection")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("ICA V/U100 single-component spectrum: task vs subject")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat-dir", type=Path, default=LAT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--max-token-fit-samples", type=int, default=0,
                        help="0 uses all projected training tokens.")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    vu_npz = args.report_dir / VU_NPZ_NAME
    ica_npz = args.report_dir / ICA_NPZ_NAME
    proj_npz = args.report_dir / PROJ_NPZ_NAME
    out_npz = args.report_dir / OUT_NPZ_NAME
    out_pdf = args.report_dir / OUT_PDF_NAME
    out_txt = args.report_dir / OUT_TXT_NAME

    files = files_for(args.lat_dir)
    V100, U100 = load_subspaces(vu_npz)
    q, scale = load_cls_query()
    token_ica, cls_ica = fit_or_load_ica(
        ica_npz, files, q, scale, V100, U100,
        max_token_fit_samples=args.max_token_fit_samples,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    ica_tokens, ica_cls, labels, subjects = compute_or_load_ica_projections(
        proj_npz, files, q, scale, V100, U100, token_ica, cls_ica
    )

    n = labels.shape[0]
    labels = labels.astype(np.int64)
    train_sids = np.arange(1, 71)
    val_sids = np.arange(71, 90)
    test_sids = np.arange(90, 110)
    m_tr = np.isin(subjects, train_sids)
    m_va = np.isin(subjects, val_sids)
    m_te = np.isin(subjects, test_sids)

    rng_split = np.random.default_rng(SEED)
    sub_split = np.empty(n, dtype=np.int8)
    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        rng_split.shuffle(idx)
        nn_ = len(idx)
        n_tr, n_va = int(0.7 * nn_), int(0.15 * nn_)
        sub_split[idx[:n_tr]] = 0
        sub_split[idx[n_tr:n_tr + n_va]] = 1
        sub_split[idx[n_tr + n_va:]] = 2
    s_tr = sub_split == 0
    s_va = sub_split == 1
    s_te = sub_split == 2
    uniq = np.unique(subjects)
    sid_to_class = {sid: i for i, sid in enumerate(uniq)}
    subject_y = np.array([sid_to_class[s] for s in subjects], dtype=np.int64)
    n_subjects = len(uniq)

    print(f"\ntask split    : tr={m_tr.sum()} va={m_va.sum()} te={m_te.sum()}", flush=True)
    print(f"subject split : tr={s_tr.sum()} va={s_va.sum()} te={s_te.sum()} "
          f"({n_subjects}-way)", flush=True)
    print("ICA fit       : V/U100 projected data, subjects 1..70 only", flush=True)
    print(f"sweeping k = 0..{K_ICA - 1} ...", flush=True)

    task_bal = np.zeros(K_ICA, dtype=np.float32)
    sub_bal = np.zeros(K_ICA, dtype=np.float32)
    t0 = time.time()
    for k in range(K_ICA):
        feat_k = np.empty((n, N_TOK + 1), dtype=np.float16)
        feat_k[:, 0] = ica_cls[:, k].astype(np.float16)
        feat_k[:, 1:] = ica_tokens[:, :, k]

        task_bal[k] = train_probe(
            feat_k[m_tr], labels[m_tr], feat_k[m_va], labels[m_va],
            feat_k[m_te], labels[m_te], 4,
        )
        sub_bal[k] = train_probe(
            feat_k[s_tr], subject_y[s_tr], feat_k[s_va], subject_y[s_va],
            feat_k[s_te], subject_y[s_te], n_subjects,
        )

        if k < 5 or k == K_ICA - 1 or k % 10 == 0:
            print(f"  k={k:>3}  task={task_bal[k]:.3f}  subject={sub_bal[k]:.3f}  "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(
        out_npz,
        k=np.arange(K_ICA),
        task_bal=task_bal,
        subject_bal=sub_bal,
        train_sids=train_sids,
        val_sids=val_sids,
        test_sids=test_sids,
        K=np.int64(K_ICA),
        seed=np.int64(SEED),
        vu_npz=str(vu_npz),
        ica_npz=str(ica_npz),
        proj_npz=str(proj_npz),
    )
    plot_results(out_pdf, task_bal, sub_bal, n_subjects)

    lines = [
        "ICA V/U100 single-component spectrum on REVE latents",
        "ICA fit: FastICA(n_components=100), fit on paper-training subjects 1..70 only",
        "Preprojection: tokens @ V_full[:, :100], CLS @ U_full[:, :100]",
        f"latents: {args.lat_dir}",
        f"task split: train={m_tr.sum()} val={m_va.sum()} test={m_te.sum()}",
        f"subject split: train={s_tr.sum()} val={s_va.sum()} test={s_te.sum()} ({n_subjects}-way)",
        f"max_token_fit_samples: {args.max_token_fit_samples} (0 means all projected training tokens)",
        "",
        "k     task_bal  sub_bal",
    ]
    for k in range(K_ICA):
        lines.append(f"{k:>4}    {task_bal[k]:.3f}    {sub_bal[k]:.3f}")
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_pdf}  {out_npz}  {out_txt}")


if __name__ == "__main__":
    main()
