
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/processed/train_index.csv")
    ap.add_argument("--out", default="data/processed/splits.csv")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    trainval, test = train_test_split(
        df, test_size=args.test_size, stratify=df["label"], random_state=args.seed
    )
    rel_val = args.val_size / (1 - args.test_size)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=trainval["label"], random_state=args.seed
    )
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    df.to_csv(args.out, index=False)

    print(df["split"].value_counts())
    print("\nClips per class per split:")
    print(pd.crosstab(df["label"], df["split"]))
    print(f"\nSaved {args.out} (seed={args.seed}). Commit this file so the team shares the same split.")


if __name__ == "__main__":
    main()