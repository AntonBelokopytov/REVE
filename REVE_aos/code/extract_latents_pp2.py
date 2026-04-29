"""Re-extract REVE latents with GLOBAL (not per-channel) z-score per recording.

Per-channel z-score destroys topography (which motor imagery decoding relies on).
The paper's stated motivation ("amplitude variations across recordings") points to
one global mu, sigma per recording, which fixes inter-recording scale differences
without normalizing channels against each other.

Pipeline:
  1. resample to 200 Hz
  2. band-pass 0.5-99.5 Hz
  3. global z-score per subject (one mu, one sigma over all channels x all samples
     of that subject's concatenated runs)
  4. clip at +/- 15

Output: /media/alex/DATA1/REVE/latents_pp2/
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
WIN_SAMPLES = 800
DATA_ROOT = Path("/media/alex/DATA1/MNE-eegbci-data/files/eegmmidb/1.0.0")
OUT_ROOT = Path("/media/alex/DATA1/REVE/latents_pp2")
BANDPASS = (0.5, 99.5)
CLIP_SIGMA = 15.0

CLASS_NAMES = ["L", "R", "B", "F"]
RUN_SCHEME = {
    3: (0, 1, 0), 4: (0, 1, 1), 5: (2, 3, 0), 6: (2, 3, 1),
    7: (0, 1, 0), 8: (0, 1, 1), 9: (2, 3, 0), 10: (2, 3, 1),
    11: (0, 1, 0), 12: (0, 1, 1), 13: (2, 3, 0), 14: (2, 3, 1),
}
BASELINE_RUNS = {1, 2}


def load_run(subject_id, run_id):
    path = DATA_ROOT / f"S{subject_id:03d}" / f"S{subject_id:03d}R{run_id:02d}.edf"
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    raw.resample(SFREQ, npad="auto", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    return raw


def events_for_run(raw, run_id):
    if run_id in BASELINE_RUNS:
        return []
    t1_label, t2_label, task_type = RUN_SCHEME[run_id]
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    inv = {v: k for k, v in event_id.items()}
    out = []
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
def encode(model, positions_1c, eeg_windows, batch_size):
    n = eeg_windows.shape[0]
    outs = []
    for i in range(0, n, batch_size):
        batch = torch.as_tensor(eeg_windows[i:i + batch_size], dtype=torch.float32)
        pos = positions_1c.unsqueeze(0).expand(batch.size(0), -1, -1)
        out = model(batch, pos)
        outs.append(out.to(torch.float16).cpu().numpy())
    return np.concatenate(outs, axis=0)


def process_subject(sid, model, pos_bank, out_dir, batch_size, overwrite):
    out_path = out_dir / f"S{sid:03d}.h5"
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name}")
        return

    runs_data = {}
    trial_events = []
    ch_names = None
    t0 = time.time()

    for run_id in range(1, 15):
        edf_path = DATA_ROOT / f"S{sid:03d}" / f"S{sid:03d}R{run_id:02d}.edf"
        if not edf_path.exists():
            continue
        try:
            raw = load_run(sid, run_id)
        except Exception as e:
            print(f"  [warn] S{sid:03d}R{run_id:02d}: {e}")
            continue
        if ch_names is None:
            ch_names = list(raw.ch_names)
        elif list(raw.ch_names) != ch_names:
            print(f"  [warn] S{sid:03d}R{run_id:02d} channel mismatch")
            continue
        eeg = raw.get_data(picks="all").astype(np.float32)
        runs_data[run_id] = eeg
        for onset, label, task_type in events_for_run(raw, run_id):
            if 0 <= onset and onset + WIN_SAMPLES <= eeg.shape[1]:
                trial_events.append((run_id, onset, label, task_type))

    if ch_names is None:
        print(f"  [skip] S{sid:03d}: no runs")
        return

    # GLOBAL z-score: one mu, one sigma over all channels and all samples
    pool = np.concatenate([runs_data[r] for r in runs_data], axis=1)  # (C, T_total)
    mu = float(pool.mean())
    sigma = float(pool.std())
    sigma = max(sigma, 1e-12)
    for r in list(runs_data):
        z = (runs_data[r] - mu) / sigma
        np.clip(z, -CLIP_SIGMA, CLIP_SIGMA, out=z)
        runs_data[r] = z.astype(np.float32)

    positions_1c = pos_bank(ch_names)
    trial_latents = None
    if trial_events:
        wins = np.stack([
            runs_data[r][:, o:o + WIN_SAMPLES] for r, o, _, _ in trial_events
        ])
        trial_latents = encode(model, positions_1c, wins, batch_size)

    tmp_path = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp_path, "w") as f:
        f.attrs["subject_id"] = sid
        f.attrs["sfreq"] = SFREQ
        f.attrs["channel_names"] = np.array(ch_names, dtype="S16")
        f.attrs["class_names"] = np.array(CLASS_NAMES, dtype="S4")
        f.attrs["bandpass_hz"] = np.array(BANDPASS, dtype=np.float64)
        f.attrs["clip_sigma"] = float(CLIP_SIGMA)
        f.attrs["zscore_mu_global"] = mu
        f.attrs["zscore_sigma_global"] = sigma
        f.attrs["zscore_mode"] = np.bytes_("global_per_subject")

        runs_grp = f.create_group("runs")
        for run_id, eeg in runs_data.items():
            g = runs_grp.create_group(f"R{run_id:02d}")
            g.create_dataset("eeg", data=eeg.astype(np.float16), compression="lzf")

        if trial_events:
            te = np.array(trial_events, dtype=np.int64)
            t = f.create_group("windows/trial")
            t.create_dataset("run",          data=te[:, 0].astype(np.uint8))
            t.create_dataset("onset_sample", data=te[:, 1].astype(np.int32))
            t.create_dataset("label",        data=te[:, 2].astype(np.int8))
            t.create_dataset("task_type",    data=te[:, 3].astype(np.uint8))
            t.create_dataset("latent",       data=trial_latents, compression="lzf")
    tmp_path.rename(out_path)

    elapsed = time.time() - t0
    n_trial = len(trial_events)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [done] S{sid:03d}: trial={n_trial} mu={mu:.2e} sigma={sigma:.2e} "
          f"file={size_mb:.1f}MB elapsed={elapsed:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subjects = args.subjects or list(range(1, 110))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"torch threads: {torch.get_num_threads()}")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).eval()
    print(f"REVE loaded. preprocessing: bandpass {BANDPASS} Hz, "
          f"GLOBAL per-subject z-score, clip ±{CLIP_SIGMA}σ")

    for sid in subjects:
        process_subject(sid, model, pos_bank, OUT_ROOT,
                        batch_size=args.batch_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
