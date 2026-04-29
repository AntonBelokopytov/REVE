"""Extract REVE latents + continuous EEG to per-subject HDF5.

Layout per subject S{XXX}.h5:
  attrs: subject_id, sfreq, channel_names, class_names
  /runs/R{RR}/eeg                (C, T)          fp16   continuous 200 Hz
  /windows/trial/run             (N,)            uint8
  /windows/trial/onset_sample    (N,)            int32
  /windows/trial/label           (N,)            int8   0..3, -1=rest
  /windows/trial/task_type       (N,)            uint8  0=movement, 1=imagery
  /windows/trial/latent          (N, C, 4, 512)  fp16   REVE output
  /windows/dense/run             (M,)            uint8  (if --dense)
  /windows/dense/onset_sample    (M,)            int32
  /windows/dense/latent          (M, C, 4, 512)  fp16
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/media/alex/DATA1/REVE/hf_cache")
_TOK = Path("/media/alex/DATA1/REVE/.hf_token")
if _TOK.exists():
    os.environ.setdefault("HF_TOKEN", _TOK.read_text().strip())

import h5py
import mne
import numpy as np
import torch
from mne.datasets import eegbci
from transformers import AutoModel

mne.set_log_level("ERROR")

SFREQ = 200
WIN_SAMPLES = 800          # 4 s
DENSE_STRIDE = 400         # 2 s
DATA_ROOT = Path("/media/alex/DATA1/MNE-eegbci-data/files/eegmmidb/1.0.0")
OUT_ROOT = Path("/media/alex/DATA1/REVE/latents")

CLASS_NAMES = ["L", "R", "B", "F"]  # 0..3; -1 = rest

# run_id -> (class for T1, class for T2, task_type: 0=movement, 1=imagery)
RUN_SCHEME = {
    3:  (0, 1, 0),  4:  (0, 1, 1),
    5:  (2, 3, 0),  6:  (2, 3, 1),
    7:  (0, 1, 0),  8:  (0, 1, 1),
    9:  (2, 3, 0), 10:  (2, 3, 1),
    11: (0, 1, 0), 12:  (0, 1, 1),
    13: (2, 3, 0), 14:  (2, 3, 1),
}
BASELINE_RUNS = {1, 2}


def load_run(subject_id: int, run_id: int) -> mne.io.BaseRaw:
    path = DATA_ROOT / f"S{subject_id:03d}" / f"S{subject_id:03d}R{run_id:02d}.edf"
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)                       # strip dots, canonical 10-05 names
    raw.resample(SFREQ, npad="auto", verbose="ERROR")
    return raw


def events_for_run(raw: mne.io.BaseRaw, run_id: int) -> list[tuple[int, int, int]]:
    """Return [(onset_sample, label, task_type), ...] for a task run."""
    if run_id in BASELINE_RUNS:
        return []
    t1_label, t2_label, task_type = RUN_SCHEME[run_id]
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    inv = {v: k for k, v in event_id.items()}
    out: list[tuple[int, int, int]] = []
    for sample, _, code in events:
        tag = inv.get(code, "")
        if tag == "T0":
            label = -1
        elif tag == "T1":
            label = t1_label
        elif tag == "T2":
            label = t2_label
        else:
            continue
        out.append((int(sample), int(label), task_type))
    return out


@torch.no_grad()
def encode(model, positions_1c: torch.Tensor, eeg_windows: np.ndarray,
           batch_size: int) -> np.ndarray:
    """eeg_windows: (N, C, WIN_SAMPLES). positions_1c: (C, 3). Returns (N, C, P, D) fp16."""
    n = eeg_windows.shape[0]
    outs: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        batch = torch.as_tensor(eeg_windows[i:i + batch_size], dtype=torch.float32)
        pos = positions_1c.unsqueeze(0).expand(batch.size(0), -1, -1)
        out = model(batch, pos)
        outs.append(out.to(torch.float16).cpu().numpy())
    return np.concatenate(outs, axis=0)


def process_subject(sid: int, model, pos_bank, out_dir: Path,
                    dense: bool, batch_size: int, overwrite: bool) -> None:
    out_path = out_dir / f"S{sid:03d}.h5"
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name} already exists")
        return

    runs_data: dict[int, np.ndarray] = {}
    trial_events: list[tuple[int, int, int, int]] = []  # (run_id, onset, label, task_type)
    ch_names: list[str] | None = None
    t0 = time.time()

    for run_id in range(1, 15):
        edf_path = DATA_ROOT / f"S{sid:03d}" / f"S{sid:03d}R{run_id:02d}.edf"
        if not edf_path.exists():
            continue
        try:
            raw = load_run(sid, run_id)
        except Exception as e:
            print(f"  [warn] S{sid:03d}R{run_id:02d} load failed: {e}")
            continue

        if ch_names is None:
            ch_names = list(raw.ch_names)
        elif list(raw.ch_names) != ch_names:
            print(f"  [warn] S{sid:03d}R{run_id:02d} channel mismatch; skipping")
            continue

        eeg = raw.get_data(picks="all").astype(np.float32)  # (C, T)
        runs_data[run_id] = eeg

        for onset, label, task_type in events_for_run(raw, run_id):
            if 0 <= onset and onset + WIN_SAMPLES <= eeg.shape[1]:
                trial_events.append((run_id, onset, label, task_type))

    if ch_names is None:
        print(f"  [skip] S{sid:03d}: no runs loaded")
        return

    positions_1c = pos_bank(ch_names)  # (C, 3)

    trial_latents: np.ndarray | None = None
    if trial_events:
        wins = np.stack([
            runs_data[r][:, o:o + WIN_SAMPLES]
            for r, o, _, _ in trial_events
        ])
        trial_latents = encode(model, positions_1c, wins, batch_size)

    dense_index: list[tuple[int, int]] = []
    dense_latents: np.ndarray | None = None
    if dense:
        dense_wins: list[np.ndarray] = []
        for run_id, eeg in runs_data.items():
            if run_id in BASELINE_RUNS:
                continue
            for o in range(0, eeg.shape[1] - WIN_SAMPLES + 1, DENSE_STRIDE):
                dense_index.append((run_id, o))
                dense_wins.append(eeg[:, o:o + WIN_SAMPLES])
        if dense_wins:
            dense_latents = encode(model, positions_1c,
                                    np.stack(dense_wins), batch_size)

    tmp_path = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp_path, "w") as f:
        f.attrs["subject_id"] = sid
        f.attrs["sfreq"] = SFREQ
        f.attrs["channel_names"] = np.array(ch_names, dtype="S16")
        f.attrs["class_names"] = np.array(CLASS_NAMES, dtype="S4")

        runs_grp = f.create_group("runs")
        for run_id, eeg in runs_data.items():
            g = runs_grp.create_group(f"R{run_id:02d}")
            g.create_dataset("eeg", data=eeg.astype(np.float16),
                             compression="lzf")

        if trial_events:
            te = np.array(trial_events, dtype=np.int64)
            t = f.create_group("windows/trial")
            t.create_dataset("run",          data=te[:, 0].astype(np.uint8))
            t.create_dataset("onset_sample", data=te[:, 1].astype(np.int32))
            t.create_dataset("label",        data=te[:, 2].astype(np.int8))
            t.create_dataset("task_type",    data=te[:, 3].astype(np.uint8))
            t.create_dataset("latent",       data=trial_latents, compression="lzf")

        if dense_index:
            de = np.array(dense_index, dtype=np.int64)
            d = f.create_group("windows/dense")
            d.create_dataset("run",          data=de[:, 0].astype(np.uint8))
            d.create_dataset("onset_sample", data=de[:, 1].astype(np.int32))
            d.create_dataset("latent",       data=dense_latents, compression="lzf")

    tmp_path.rename(out_path)

    elapsed = time.time() - t0
    n_trial = len(trial_events)
    n_dense = len(dense_index)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [done] S{sid:03d}: runs={sorted(runs_data)} "
          f"trial={n_trial} dense={n_dense} file={size_mb:.1f} MB "
          f"elapsed={elapsed:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", type=int,
                    help="subject IDs (1..109). Default: all.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dense", action="store_true",
                    help="also extract dense sliding-window latents (slow on CPU)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subjects = args.subjects or list(range(1, 110))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"torch threads: {torch.get_num_threads()}")
    print("Loading REVE...")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    print(f"REVE loaded. dense={args.dense} batch_size={args.batch_size}")

    for sid in subjects:
        process_subject(sid, model, pos_bank, OUT_ROOT,
                        dense=args.dense, batch_size=args.batch_size,
                        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
