"""Blend model probability outputs and write the final submission.

Discovers oof_<tag>.npy / test_<tag>.npy pairs under oof/ (downloaded from Kaggle
kernels, plus the local feature model), reports each model's OOF balanced macro,
learns non-negative blend weights that maximize the OOF balanced macro, and writes
submission.csv. All arrays are aligned to train.csv / test.csv row order.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clinspan.data import load, LETTERS  # noqa
from clinspan.metric import balanced_macro_accuracy, per_group_accuracy  # noqa

OOF = ROOT / "oof"
SUBM = ROOT / "submissions"
SUBM.mkdir(exist_ok=True)


def normalize(p):
    p = np.clip(p, 1e-6, None)
    return p / p.sum(1, keepdims=True)


def make_fast_scorer(train):
    """Fast balanced-macro scorer over probability matrices (no pandas in the loop)."""
    y = train["selected_option"].map({l: i for i, l in enumerate(LETTERS)}).to_numpy()
    grp = (train["target_language"].astype(str) + "|" + train["entity_type"].astype(str))
    g = grp.astype("category").cat.codes.to_numpy()
    ng = g.max() + 1
    counts = np.bincount(g, minlength=ng).astype(float)

    def scorer(probs):
        pred = probs.argmax(1)
        correct = (pred == y).astype(float)
        per = np.bincount(g, weights=correct, minlength=ng) / np.maximum(counts, 1)
        return float(per.mean())

    return scorer


def discover():
    tags = []
    for f in sorted(glob.glob(str(OOF / "oof_*.npy"))):
        tag = Path(f).stem[len("oof_"):]
        tf = OOF / f"test_{tag}.npy"
        if tf.exists():
            tags.append(tag)
    # feature model uses feat_oof/feat_test naming
    if (OOF / "feat_oof.npy").exists() and (OOF / "feat_test.npy").exists():
        tags.append("feat")
    return tags


def load_arr(tag, kind):
    if tag == "feat":
        return np.load(OOF / f"{'feat_oof' if kind=='oof' else 'feat_test'}.npy")
    return np.load(OOF / f"{kind}_{tag}.npy")


def optimize_weights(scorer, oofs, seed=0, iters=30000):
    """Random search on the simplex + coordinate refine to maximize OOF balanced macro."""
    rng = np.random.default_rng(seed)
    n = len(oofs)
    stack = np.stack([normalize(o) for o in oofs])  # (n, N, 4)

    def blended(w):
        return (stack * np.asarray(w)[:, None, None]).sum(0)

    best_w = np.ones(n) / n
    best = scorer(blended(best_w))
    for _ in range(iters):
        w = rng.dirichlet(np.ones(n) * 0.7)
        s = scorer(blended(w))
        if s > best:
            best, best_w = s, w
    grid = np.linspace(0, 1, 41)
    for _ in range(4):
        for i in range(n):
            base = best_w.copy()
            for gv in grid:
                w = base.copy()
                w[i] = gv
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                s = scorer(blended(w))
                if s > best:
                    best, best_w = s, w
    return best_w / best_w.sum(), best


def main():
    train, test = load()
    tags = discover()
    if not tags:
        print("No model outputs found in", OOF)
        return
    print("models:", tags)

    scorer = make_fast_scorer(train)
    oofs, tests = [], []
    print("\n=== per-model OOF balanced macro ===")
    for t in tags:
        o = load_arr(t, "oof")
        te = load_arr(t, "test")
        assert o.shape == (len(train), 4), f"{t} oof shape {o.shape}"
        assert te.shape == (len(test), 4), f"{t} test shape {te.shape}"
        oofs.append(o)
        tests.append(te)
        print(f"  {t:14s}: {scorer(normalize(o)):.4f}")

    w, best = optimize_weights(scorer, oofs)
    print("\n=== blend ===")
    print("weights:", {t: round(float(wi), 3) for t, wi in zip(tags, w)})
    print(f"blend OOF balanced macro: {best:.4f}")

    stack_oof = np.stack([normalize(o) for o in oofs])
    stack_test = np.stack([normalize(o) for o in tests])
    blend_oof = (stack_oof * w[:, None, None]).sum(0)
    blend_test = (stack_test * w[:, None, None]).sum(0)

    train2 = train.copy()
    train2["prediction"] = [LETTERS[i] for i in blend_oof.argmax(1)]
    print("\n=== blend per-group OOF ===")
    print(per_group_accuracy(train2).to_string(index=False))

    sub = pd.DataFrame({"id": test["id"],
                        "selected_option": [LETTERS[i] for i in blend_test.argmax(1)]})
    out = SUBM / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(sub)} rows)")
    print(sub["selected_option"].value_counts().to_dict())
    np.save(OOF / "blend_test.npy", blend_test)
    np.save(OOF / "blend_oof.npy", blend_oof)


if __name__ == "__main__":
    main()
