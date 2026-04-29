"""Re-extract REVE latents using the EXACT preprocessing from reve_eeg/preprocessing/preprocessing_physio.py.

Pipeline per the official repo:
  1. raw.set_eeg_reference("average")        # CAR
  2. raw.filter(l_freq=0.3, h_freq=None)     # 0.3 Hz highpass only
  3. raw.notch_filter(60)                    # 60 Hz notch
  4. raw.resample(200)                       # 200 Hz
  5. mne.Epochs(tmin=0, tmax=4-1/sfreq)
  6. data = epochs.get_data(units="uV")[:, :, -800:]   # microvolts, last 800 samples
  7. (at training time) sample / 100         # scale_factor=100

Tasks: 04, 06, 08, 10, 12, 14 (imagery only). T0=rest is dropped.
Labels: 04/08/12: T1=L=0, T2=R=1.   06/10/14: T1=B=2, T2=F=3.

Output one HDF5 per subject in /media/alex/DATA1/REVE/latents_paper/, with:
  /trial/latent  (N, 64, 4, 512) fp16   REVE output for (uV/100) input
  /trial/label   (N,)            int8
  /trial/run     (N,)            uint8
  /trial/event_idx (N,)          int32  index within original epoch list
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
SCALE_FACTOR = 100.0    # model input = uV / 100, per LMDBDataset
DATA_ROOT = Path("/media/alex/DATA1/MNE-eegbci-data/files/eegmmidb/1.0.0")
OUT_ROOT = Path("/media/alex/DATA1/REVE/latents_paper")
TASKS = [4, 6, 8, 10, 12, 14]
LR_TASKS = {4, 8, 12}      # T1=L=0, T2=R=1
BF_TASKS = {6, 10, 14}     # T1=B=2, T2=F=3


def load_and_preprocess(subject_id: int, run_id: int) -> mne.io.BaseRaw:
    path = DATA_ROOT / f"S{subject_id:03d}" / f"S{subject_id:03d}R{run_id:02d}.edf"
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)                                   # rename channels (in-place)
    if len(raw.info["bads"]) > 0:
        raw.interpolate_bads()
    raw.set_eeg_reference(ref_channels="average", verbose="ERROR")
    raw.filter(l_freq=0.3, h_freq=None, fir_design="firwin", verbose="ERROR")
    raw.notch_filter(60.0, verbose="ERROR")
    raw.resample(SFREQ, npad="auto", verbose="ERROR")
    return raw


def extract_run_trials(raw: mne.io.BaseRaw, run_id: int):
    """Returns (data_uV (N,C,800) fp32, labels (N,) int8, ev_idx (N,) int32) or empty."""
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if events.size == 0:
        return None
    sf = raw.info["sfreq"]
    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=0.0, tmax=4.0 - 1.0 / sf,
        baseline=None, preload=True, verbose="ERROR",
    )
    data = epochs.get_data(units="uV")           # (n_ep, C, T)  in uV
    if data.shape[2] < WIN_SAMPLES:
        return None
    data = data[:, :, -WIN_SAMPLES:]              # last 800 samples
    ev = epochs.events[:, 2]                      # 1, 2, or 3 (T0/T1/T2)
    keep = ev != 1                                # drop rest
    if not keep.any():
        return None
    data = data[keep]
    ev = ev[keep]
    ev_idx = np.where(keep)[0].astype(np.int32)
    if run_id in LR_TASKS:
        labels = (ev - 2).astype(np.int8)         # 2->0 (L), 3->1 (R)
    elif run_id in BF_TASKS:
        labels = ev.astype(np.int8)               # 2->2 (B), 3->3 (F)
    else:
        return None
    return data.astype(np.float32), labels, ev_idx


@torch.no_grad()
def encode(model, positions_1c, eeg_uV, batch_size):
    """eeg_uV: (N, C, 800) in uV. Returns (N, C, P, D) fp16."""
    n = eeg_uV.shape[0]
    outs = []
    for i in range(0, n, batch_size):
        batch_uV = eeg_uV[i:i + batch_size]
        batch_in = batch_uV / SCALE_FACTOR        # the head sees uV / 100
        x = torch.as_tensor(batch_in, dtype=torch.float32)
        pos = positions_1c.unsqueeze(0).expand(x.size(0), -1, -1)
        out = model(x, pos)                       # (B, C, P, D)
        outs.append(out.to(torch.float16).cpu().numpy())
    return np.concatenate(outs, axis=0)


def process_subject(sid, model, pos_bank, out_dir, batch_size, overwrite):
    out_path = out_dir / f"S{sid:03d}.h5"
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name}")
        return

    t0 = time.time()
    all_data, all_labels, all_runs, all_evidx = [], [], [], []
    ch_names = None

    for run_id in TASKS:
        edf_path = DATA_ROOT / f"S{sid:03d}" / f"S{sid:03d}R{run_id:02d}.edf"
        if not edf_path.exists():
            continue
        try:
            raw = load_and_preprocess(sid, run_id)
        except Exception as e:
            print(f"  [warn] S{sid:03d}R{run_id:02d}: {e}")
            continue
        if ch_names is None:
            ch_names = list(raw.ch_names)
        elif list(raw.ch_names) != ch_names:
            print(f"  [warn] S{sid:03d}R{run_id:02d}: channel mismatch")
            continue
        result = extract_run_trials(raw, run_id)
        if result is None:
            continue
        data, labels, ev_idx = result
        all_data.append(data); all_labels.append(labels)
        all_runs.append(np.full(data.shape[0], run_id, dtype=np.uint8))
        all_evidx.append(ev_idx)

    if not all_data:
        print(f"  [skip] S{sid:03d}: no trials")
        return

    eeg_uV = np.concatenate(all_data)         # (N, C, 800) uV
    labels = np.concatenate(all_labels)
    runs = np.concatenate(all_runs)
    evidx = np.concatenate(all_evidx)

    positions_1c = pos_bank(ch_names)
    latents = encode(model, positions_1c, eeg_uV, batch_size)

    tmp_path = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp_path, "w") as f:
        f.attrs["subject_id"] = sid
        f.attrs["sfreq"] = SFREQ
        f.attrs["channel_names"] = np.array(ch_names, dtype="S16")
        f.attrs["scale_factor"] = SCALE_FACTOR
        f.attrs["preprocessing"] = np.bytes_(
            "CAR + 0.3Hz HP + 60Hz notch + resample 200Hz; epochs uV; input=uV/100"
        )
        g = f.create_group("trial")
        g.create_dataset("latent", data=latents, compression="lzf")
        g.create_dataset("label", data=labels)
        g.create_dataset("run", data=runs)
        g.create_dataset("event_idx", data=evidx)
    tmp_path.rename(out_path)

    elapsed = time.time() - t0
    n_trial = labels.size
    size_mb = out_path.stat().st_size / 1e6
    cls_counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"  [done] S{sid:03d}: trials={n_trial} classes={cls_counts} "
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
    print("REVE loaded. preprocessing matches reve_eeg/preprocessing/preprocessing_physio.py.")

    for sid in subjects:
        process_subject(sid, model, pos_bank, OUT_ROOT,
                        batch_size=args.batch_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
