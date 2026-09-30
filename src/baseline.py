
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

GRIDS = {
    "svm": [{"C": c} for c in (1, 10, 100)],
    "rf": [{"n_estimators": n} for n in (200, 500)],
    "logreg": [{"C": c} for c in (0.1, 1, 10)],
}


def build(model, params):
    if model == "svm":
        clf = SVC(kernel="rbf", gamma="scale", **params)
    elif model == "rf":
        clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    else:
        clf = LogisticRegression(max_iter=3000, **params)
    return make_pipeline(StandardScaler(), clf)


def aggregate(X):
    """(N, T, F) -> (N, 4F): mean, std, min, max over time for every coefficient."""
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="data/processed/features/w2_energy_mfcc40")
    ap.add_argument("--model", choices=list(GRIDS), default="svm")
    ap.add_argument("--fig-dir", default="reports/figures")
    ap.add_argument("--log", default="reports/experiments.csv")
    args = ap.parse_args()

    fd = Path(args.feat_dir)
    tr, va, te = (np.load(fd / f"{s}.npz", allow_pickle=True) for s in ("train", "val", "test"))
    label2id = json.loads((fd / "labels.json").read_text())
    names = [k for k, _ in sorted(label2id.items(), key=lambda kv: kv[1])]

    Xtr, Xva, Xte = aggregate(tr["X"]), aggregate(va["X"]), aggregate(te["X"])
    ytr, yva, yte = tr["y"], va["y"], te["y"]
    print(f"Aggregated feature shape: {Xtr.shape}")

    best = None
    for params in GRIDS[args.model]:
        t0 = time.time()
        pipe = build(args.model, params).fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        f1 = f1_score(yva, pred, average="macro")
        acc = accuracy_score(yva, pred)
        print(f"{args.model} {params}: val acc={acc:.4f} macro-F1={f1:.4f} ({time.time()-t0:.1f}s)")
        if best is None or f1 > best["f1"]:
            best = {"params": params, "f1": f1, "acc": acc, "train_s": time.time() - t0}

    model = build(args.model, best["params"]).fit(Xtr, ytr)
    t0 = time.time()
    pred = model.predict(Xte)
    infer_s = time.time() - t0
    test_acc = accuracy_score(yte, pred)
    test_f1 = f1_score(yte, pred, average="macro")
    print(f"\nBest params: {best['params']}")
    print(f"TEST accuracy={test_acc:.4f} macro-F1={test_f1:.4f}")
    print(classification_report(yte, pred, target_names=names, digits=3))

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        yte, pred, display_labels=names, xticks_rotation=45, ax=ax, colorbar=False
    )
    ax.set_title(f"Baseline {args.model} - test confusion matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / f"cm_baseline_{args.model}.png", dpi=150)
    plt.close()

    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": f"baseline_{args.model}",
        "features": fd.name,
        "params": json.dumps(best["params"]),
        "val_acc": round(best["acc"], 4),
        "val_macro_f1": round(best["f1"], 4),
        "test_acc": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "train_seconds": round(best["train_s"], 1),
        "inference_seconds": round(infer_s, 2),
    }
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"\nLogged run to {log}")


if __name__ == "__main__":
    main()