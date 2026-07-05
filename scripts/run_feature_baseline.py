"""Gradient-boosting baseline over pairwise (anchor, candidate) features.

Trains a binary LightGBM on the 4-way-expanded rows (P(option correct)), does
grouped 5-fold CV, argmaxes over the four options per example, and reports the
official balanced macro accuracy. Saves OOF + test probabilities for ensembling.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinspan.data import load, add_folds, CANDS, LETTERS
from clinspan.features import build_pairwise, feature_columns
from clinspan.metric import balanced_macro_accuracy, per_group_accuracy

OOF_DIR = ROOT / "oof"
OOF_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
SEED = 42


def encode_cats(df):
    df = df.copy()
    df["lang_code"] = df["target_language"].astype("category").cat.codes
    df["ent_code"] = df["entity_type"].astype("category").cat.codes
    return df


def main():
    train, test = load()
    train = add_folds(train, n_splits=N_SPLITS, seed=SEED)

    print("Building pairwise features (train)...")
    tr_pair = build_pairwise(train)
    print("Building pairwise features (test)...")
    te_pair = build_pairwise(test)
    tr_pair = encode_cats(tr_pair)
    te_pair = encode_cats(te_pair)

    feat_cols = feature_columns(tr_pair) + ["lang_code", "ent_code"]
    feat_cols = [c for c in feat_cols if c not in ("lang_code", "ent_code")] + ["lang_code", "ent_code"]

    # map fold to each expanded row via the source row index
    row_fold = train["fold"].to_numpy()
    tr_pair["fold"] = row_fold[tr_pair["row"].to_numpy()]

    params = dict(
        objective="binary", metric="binary_logloss", learning_rate=0.03,
        num_leaves=31, min_child_samples=40, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
        verbose=-1, n_jobs=-1, seed=SEED,
    )

    oof_prob = np.zeros(len(tr_pair))
    test_prob = np.zeros(len(te_pair))
    for fold in range(N_SPLITS):
        tr_idx = tr_pair["fold"] != fold
        va_idx = tr_pair["fold"] == fold
        dtr = lgb.Dataset(tr_pair.loc[tr_idx, feat_cols], label=tr_pair.loc[tr_idx, "y"])
        dva = lgb.Dataset(tr_pair.loc[va_idx, feat_cols], label=tr_pair.loc[va_idx, "y"])
        model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        oof_prob[va_idx.to_numpy()] = model.predict(tr_pair.loc[va_idx, feat_cols])
        test_prob += model.predict(te_pair[feat_cols]) / N_SPLITS
        print(f"  fold {fold}: best_iter={model.best_iteration}")

    # reshape to (n_examples, 4)
    def to_matrix(pair_df, prob, n_rows):
        mat = np.zeros((n_rows, 4))
        rows = pair_df["row"].to_numpy()
        opts = pair_df["opt"].to_numpy()
        mat[rows, opts] = prob
        return mat

    oof_mat = to_matrix(tr_pair, oof_prob, len(train))
    test_mat = to_matrix(te_pair, test_prob, len(test))

    train["prediction"] = [LETTERS[i] for i in oof_mat.argmax(1)]
    score = balanced_macro_accuracy(train)
    plain = (train["prediction"] == train["selected_option"]).mean()
    print(f"\n=== Feature baseline ===")
    print(f"balanced macro accuracy (OOF): {score:.4f}")
    print(f"plain accuracy (OOF):          {plain:.4f}")
    print(per_group_accuracy(train).to_string(index=False))

    np.save(OOF_DIR / "feat_oof.npy", oof_mat)
    np.save(OOF_DIR / "feat_test.npy", test_mat)
    train[["id", "fold"]].to_csv(OOF_DIR / "folds.csv", index=False)
    print(f"\nSaved OOF/test probs to {OOF_DIR}")

    # feature importance
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=feat_cols)
    print("\nTop features by gain:")
    print(imp.sort_values(ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
