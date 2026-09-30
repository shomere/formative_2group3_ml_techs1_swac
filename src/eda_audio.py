
import argparse
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 16000


def clip_stats(path, top_db=30):
    y, _ = librosa.load(path, sr=SR, mono=True)
    intervals = librosa.effects.split(y, top_db=top_db)
    active = sum(e - s for s, e in intervals) / SR
    return {
        "active_sec": active,
        "n_segments": len(intervals),
        "rms": float(np.sqrt(np.mean(y**2))),
        "peak": float(np.max(np.abs(y))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/processed/train_index.csv")
    ap.add_argument("--fig-dir", default="reports/figures")
    ap.add_argument("--out", default="data/processed/audio_stats.csv")
    ap.add_argument("--top-db", type=float, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    stats = [clip_stats(p, args.top_db) for p in tqdm(df["path"], desc="Analysing clips")]
    df = pd.concat([df, pd.DataFrame(stats)], axis=1)
    df["speech_frac"] = df["active_sec"] / df["duration"]
    df.to_csv(args.out, index=False)

    print("\nLongest 10 clips:")
    print(df.nlargest(10, "duration")[["id", "label", "duration", "active_sec", "n_segments"]])
    for t in (6, 8, 10, 15):
        print(f"clips longer than {t}s: {(df['duration'] > t).sum()}")
    print("\nActive speech seconds:")
    print(df["active_sec"].describe())
    print("\nSpeech fraction:")
    print(df["speech_frac"].describe())
    print("\nSegments per clip:")
    print(df["n_segments"].value_counts().sort_index().head(10))
    print("\nPeak amplitude (near 1.0 = clipping risk):")
    print(df["peak"].describe())
    print("clips with peak >= 0.99:", (df["peak"] >= 0.99).sum())

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.scatter(df["duration"], df["active_sec"], s=4, alpha=0.4)
    plt.xlabel("total duration (s)")
    plt.ylabel("active speech (s)")
    plt.title("Total vs active speech duration")
    plt.tight_layout()
    plt.savefig(fig_dir / "speech_vs_total.png", dpi=150)
    plt.close()

    df.boxplot(column="duration", by="label", figsize=(10, 4), rot=45)
    plt.suptitle("")
    plt.title("Duration by class")
    plt.tight_layout()
    plt.savefig(fig_dir / "duration_by_class.png", dpi=150)
    plt.close()

    labels = sorted(df["label"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, lab in zip(axes.ravel(), labels):
        row = df[df["label"] == lab].iloc[0]
        y, _ = librosa.load(row["path"], sr=SR)
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64), ref=np.max)
        ax.imshow(m, origin="lower", aspect="auto")
        ax.set_title(f"{lab} ({row['duration']:.1f}s)")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(fig_dir / "examples_mel.png", dpi=150)
    plt.close()
    print(f"\nSaved figures to {fig_dir} and stats to {args.out}")


if __name__ == "__main__":
    main()