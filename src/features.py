
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

SR = 16000
FRAME = 400   # 25 ms
HOP = 160     # 10 ms


def select_window(y, width, mode):
    """Return exactly `width` samples: pad short clips, crop long ones."""
    if len(y) <= width:
        total = width - len(y)
        left = total // 2
        return np.pad(y, (left, total - left))
    if mode == "start":
        return y[:width]
    # 'energy': slide a window and keep the most energetic region
    e = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0] ** 2
    win = width // HOP
    csum = np.concatenate([[0.0], np.cumsum(e)])
    scores = csum[win:] - csum[: len(csum) - win]
    start = int(np.argmax(scores)) * HOP
    seg = y[start : start + width]
    if len(seg) < width:
        seg = np.pad(seg, (0, width - len(seg)))
    return seg


def clip_to_mfcc(path, width, mode, n_mfcc, deltas):
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak                      # peak-normalise (peaks range from 0.005 to 1.0)
    y = select_window(y, width, mode)
    m = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=n_mfcc, n_fft=FRAME, hop_length=HOP, n_mels=64
    )
    if deltas:
        m = np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)])
    return m.T.astype(np.float32)         # (time, features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/processed/splits.csv")
    ap.add_argument("--out-root", default="data/processed/features")
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--select", choices=["energy", "start"], default="energy")
    ap.add_argument("--n-mfcc", type=int, default=40)
    ap.add_argument("--deltas", action="store_true", help="append delta and delta-delta")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    tag = args.tag or f"w{args.window_sec:g}_{args.select}_mfcc{args.n_mfcc}" + (
        "_d" if args.deltas else ""
    )
    out_dir = Path(args.out_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.splits)
    classes = sorted(df["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    (out_dir / "labels.json").write_text(json.dumps(label2id, indent=2))
    width = int(args.window_sec * SR)

    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        feats = Parallel(n_jobs=args.n_jobs)(
            delayed(clip_to_mfcc)(p, width, args.select, args.n_mfcc, args.deltas)
            for p in tqdm(part["path"], desc=split)
        )
        X = np.stack(feats)
        y = part["label"].map(label2id).to_numpy()
        np.savez_compressed(out_dir / f"{split}.npz", X=X, y=y, ids=part["id"].to_numpy())
        print(f"{split}: X{X.shape} y{y.shape}")
    print(f"\nSaved features to {out_dir}")


if __name__ == "__main__":
    main()