"""Enhanced structural GBM: bipartite token matching between the anchor and each
candidate over *structural* similarity (length, star-mask shape, visible-char
alignment, digit patterns) — no TF-IDF / term-frequency statistics.

The redaction preserves morpheme SHAPE, punctuation, numbers, and length. Cross-
lingual clinical cognates therefore keep aligned length + visible-letter skeletons
even after redaction; an order-agnostic optimal token assignment captures this.

Outputs oof_struct.npy / test_struct.npy for the ensemble.
"""
import re, sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clinspan.data import load, add_folds, CANDS, LETTERS, anchor_span
from clinspan.metric import balanced_macro_accuracy, per_group_accuracy

OOF = ROOT / "oof"; OOF.mkdir(exist_ok=True)
N_SPLITS, SEED = 5, 42
_TOK = re.compile(r"[^\s]+")


def mask(t):
    return "".join("L" if c.isalpha() else "#" if c.isdigit() else "*" if c == "*" else "." for c in t)


def visible(t):
    return [c for c in t.lower() if c.isalpha()]


def digits(t):
    return set(re.findall(r"\d", t))


def tok_sim(a, c):
    """Structural similarity of two redacted tokens in [0,1]."""
    la, lc = len(a), len(c)
    if la == 0 or lc == 0:
        return 0.0
    len_score = 1.0 - abs(la - lc) / max(la, lc)
    va, vc = Counter(visible(a)), Counter(visible(c))
    inter = sum((va & vc).values()); union = sum((va | vc).values())
    vis = inter / union if union else 0.0
    shape = SequenceMatcher(None, mask(a), mask(c)).ratio()
    # position-aligned visible-char agreement (redaction keeps length -> positions line up)
    pos = 0.0
    if la == lc:
        agree = sum(1 for x, y in zip(a.lower(), c.lower()) if x == y and x.isalpha())
        vis_positions = sum(1 for x in a.lower() if x.isalpha())
        pos = agree / vis_positions if vis_positions else 0.0
    da, dc = digits(a), digits(c)
    dig = len(da & dc) / len(da | dc) if (da | dc) else 0.0
    return 0.34 * vis + 0.22 * len_score + 0.24 * shape + 0.12 * pos + 0.08 * dig


def tokens(s):
    return _TOK.findall(str(s))


def bipartite_feats(anchor, cand):
    A, C = tokens(anchor), tokens(cand)
    f = {}
    if not A or not C:
        return {"bm_sum": 0, "bm_mean": 0, "bm_max": 0, "bm_nstrong": 0,
                "bm_cov": 0, "bm_lw": 0, "na": len(A), "nc": len(C), "bm_min": 0}
    S = np.array([[tok_sim(a, c) for c in C] for a in A])  # (|A|,|C|)
    ri, ci = linear_sum_assignment(-S)
    matched = S[ri, ci]
    lw = sum(matched[k] * len(A[ri[k]]) for k in range(len(ri))) / (sum(len(a) for a in A) + 1e-6)
    f.update(bm_sum=float(matched.sum()), bm_mean=float(matched.mean()),
             bm_max=float(matched.max()), bm_min=float(matched.min()),
             bm_nstrong=int((matched > 0.5).sum()), bm_cov=len(matched) / max(len(A), len(C)),
             bm_lw=float(lw), na=len(A), nc=len(C))
    return f


def global_feats(anchor, cand):
    a, c = str(anchor), str(cand)
    la, lc = len(a), len(c)
    va, vc = Counter(visible(a)), Counter(visible(c))
    inter = sum((va & vc).values()); union = sum((va | vc).values())
    da, dc = digits(a), digits(c)
    return {
        "len_ratio": lc / (la + 1e-6), "len_absdiff": abs(la - lc),
        "vis_jac": inter / union if union else 0.0,
        "vis_overlap": inter / (min(sum(va.values()), sum(vc.values())) + 1e-6),
        "star_frac_a": a.count("*") / (la + 1e-6), "star_frac_c": c.count("*") / (lc + 1e-6),
        "dig_jac": len(da & dc) / len(da | dc) if (da | dc) else 0.0,
        "mask_ratio": SequenceMatcher(None, mask(a), mask(c)).ratio(),
        "seq_ratio": SequenceMatcher(None, a.lower(), c.lower()).ratio(),
        "ntok_ratio": len(tokens(c)) / (len(tokens(a)) + 1e-6),
    }


def build(df):
    recs = []
    for idx, r in df.iterrows():
        anchor, left, right = anchor_span(r["source_context"])
        loc = " ".join(left.split()[-5:] + right.split()[:5])
        per = []
        for oi, col in enumerate(CANDS):
            cand = str(r[col])
            d = {}
            d.update(bipartite_feats(anchor, cand))
            d.update(global_feats(anchor, cand))
            # local Spanish context around anchor vs candidate (weak cross-lingual cue, structural)
            d["ctx_bm"] = bipartite_feats(loc, cand)["bm_mean"]
            per.append(d)
        # within-row relative features on the key structural scores
        for key in ["bm_sum", "bm_mean", "bm_max", "vis_jac", "seq_ratio", "len_ratio", "bm_lw", "vis_overlap"]:
            vals = np.array([p[key] for p in per], float)
            ranks = vals.argsort().argsort()
            mx = vals.max()
            for oi, p in enumerate(per):
                others = np.delete(vals, oi)
                p[f"{key}_rank"] = float(ranks[oi])
                p[f"{key}_margin"] = float(vals[oi] - others.max())
                p[f"{key}_ismax"] = float(vals[oi] == mx and mx > 0)
        for oi, p in enumerate(per):
            rec = {"row": idx, "opt": oi, "target_language": r["target_language"],
                   "entity_type": r["entity_type"]}
            rec.update(p)
            if "selected_option" in r:
                rec["y"] = int("ABCD"[oi] == r["selected_option"])
            recs.append(rec)
    return pd.DataFrame.from_records(recs)


def main():
    train, test = load()
    train = add_folds(train, n_splits=N_SPLITS, seed=SEED)
    print("building structural features (train)...")
    trp = build(train)
    print("building structural features (test)...")
    tep = build(test)
    for d in (trp, tep):
        d["lang_code"] = d["target_language"].astype("category").cat.codes
        d["ent_code"] = d["entity_type"].astype("category").cat.codes
    excl = {"row", "opt", "target_language", "entity_type", "y", "fold"}
    feat_cols = [c for c in trp.columns if c not in excl]
    trp["fold"] = train["fold"].to_numpy()[trp["row"].to_numpy()]

    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.02,
                  num_leaves=64, min_child_samples=50, feature_fraction=0.6,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, verbose=-1, n_jobs=-1, seed=SEED)
    oof_p = np.zeros(len(trp)); test_p = np.zeros(len(tep)); models = []
    for fold in range(N_SPLITS):
        tr, va = trp["fold"] != fold, trp["fold"] == fold
        m = lgb.train(params, lgb.Dataset(trp.loc[tr, feat_cols], trp.loc[tr, "y"]),
                      num_boost_round=4000, valid_sets=[lgb.Dataset(trp.loc[va, feat_cols], trp.loc[va, "y"])],
                      callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
        oof_p[va.to_numpy()] = m.predict(trp.loc[va, feat_cols])
        test_p += m.predict(tep[feat_cols]) / N_SPLITS
        models.append(m)

    def to_mat(pair, p, n):
        mat = np.zeros((n, 4)); mat[pair["row"].to_numpy(), pair["opt"].to_numpy()] = p; return mat
    oof_m, test_m = to_mat(trp, oof_p, len(train)), to_mat(tep, test_p, len(test))
    train["prediction"] = [LETTERS[i] for i in oof_m.argmax(1)]
    print("\n=== structural GBM OOF balanced macro:", round(balanced_macro_accuracy(train), 4), "===")
    print(per_group_accuracy(train).to_string(index=False))
    np.save(OOF / "oof_struct.npy", oof_m); np.save(OOF / "test_struct.npy", test_m)
    imp = pd.Series(models[-1].feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    print("\ntop features:\n", imp.head(15).to_string())
    print("saved oof_struct / test_struct")


if __name__ == "__main__":
    main()
