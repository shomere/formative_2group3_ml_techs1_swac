
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def detect_columns(df, id_col=None, label_col=None):
    cols = list(df.columns)
    if id_col is None:
        id_col = next((c for c in cols if "id" in c.lower()), cols[0])
    if label_col is None:
        label_col = next(
            (c for c in cols if c.lower() in {"label", "target", "class"}),
            [c for c in cols if c != id_col][-1],
        )
    return id_col, label_col


def index_audio(audio_dir):
    """Map file stem -> path for every audio file under audio_dir."""
    files = {}
    for p in Path(audio_dir).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            files[p.stem] = p
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/raw/Train.csv")
    ap.add_argument("--audio-dir", default="data/raw")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--fig-dir", default="reports/figures")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    print("Columns:", list(df.columns))
    print(df.head(), "\n")
    id_col, label_col = detect_columns(df, args.id_col, args.label_col)
    print(f"Using id column='{id_col}', label column='{label_col}'")

    audio = index_audio(args.audio_dir)
    print(f"Audio files found under {args.audio_dir}: {len(audio)}")

    df["_stem"] = df[id_col].astype(str).map(lambda s: Path(s).stem)
    df["path"] = df["_stem"].map(audio)
    missing = df["path"].isna().sum()
    print(f"Train rows: {len(df)} | rows with no matching audio: {missing}")
    df = df.dropna(subset=["path"]).copy()

    durations, rates, chans = [], [], []
    for p in tqdm(df["path"], desc="Reading audio headers"):
        info = sf.info(str(p))
        durations.append(info.duration)
        rates.append(info.samplerate)
        chans.append(info.channels)
    df["duration"], df["sample_rate"], df["channels"] = durations, rates, chans

    print("\nClass distribution:")
    print(df[label_col].value_counts())
    print("\nDuration (s) summary:")
    print(df["duration"].describe())
    print("\nSample rates:", df["sample_rate"].value_counts().to_dict())
    print("Channels:", df["channels"].value_counts().to_dict())
    print("\nMean duration per class (s):")
    print(df.groupby(label_col)["duration"].mean().round(2))

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = df.rename(columns={id_col: "id", label_col: "label"})[
        ["id", "label", "path", "duration", "sample_rate", "channels"]
    ]
    out.to_csv(out_dir / "train_index.csv", index=False)

    ax = out["label"].value_counts().plot(kind="bar", figsize=(9, 4))
    ax.set_title("Class distribution")
    ax.set_ylabel("clips")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_counts.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(out["duration"], bins=40)
    plt.xlabel("duration (s)")
    plt.ylabel("clips")
    plt.title("Clip duration distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "duration_hist.png", dpi=150)
    plt.close()
    print(f"\nSaved index to {out_dir/'train_index.csv'} and figures to {fig_dir}")


if __name__ == "__main__":
    main()