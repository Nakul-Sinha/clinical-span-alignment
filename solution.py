import re
import sys
import random
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
DATA = Path(r"G:\Datacurve\clinical chal\dataset")
OOF = ROOT / "oof"
SUB = ROOT / "submissions"
SUB.mkdir(exist_ok=True)

CANDS = ["candidate_a", "candidate_b", "candidate_c", "candidate_d"]
LETTERS = ["A", "B", "C", "D"]
GROUPS = ["target_language", "entity_type"]
ANCHOR = re.compile(r"\[\[ANCHOR\]\](.*?)\[\[/ANCHOR\]\]", re.DOTALL)
TOK = re.compile(r"[^\s]+")
N_SPLITS = 5
SEED = 42
TRANSFORMER_TAGS = ["fb2", "hwbase", "mdeberta", "xlmr-base", "lgbase"]


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        df["anchor"] = df["source_context"].map(lambda s: (ANCHOR.search(s).group(1).strip() if ANCHOR.search(s) else ""))
    return train, test


def anchor_span(s):
    m = ANCHOR.search(s)
    if not m:
        return "", s, ""
    return m.group(1).strip(), s[:m.start()], s[m.end():]


def add_folds(df, n=N_SPLITS, seed=SEED):
    import hashlib
    h = lambda x: int(hashlib.md5(f"{seed}:{x}".encode()).hexdigest(), 16)
    df = df.copy()
    df["_h"] = df["id"].map(h)
    df["fold"] = -1
    for _, idx in df.groupby(GROUPS).groups.items():
        sub = df.loc[idx].sort_values("_h")
        df.loc[sub.index, "fold"] = np.arange(len(sub)) % n
    return df["fold"].to_numpy()


def y_index(df):
    return df["selected_option"].map({l: i for i, l in enumerate(LETTERS)}).to_numpy()


def balanced_macro(df, pred_idx):
    df = df.copy()
    df["_ok"] = pred_idx == y_index(df)
    return float(np.mean([g["_ok"].mean() for _, g in df.groupby(GROUPS)]))


def per_group(df, pred_idx):
    df = df.copy()
    df["_ok"] = pred_idx == y_index(df)
    rows = [{"lang": l, "ent": e, "n": len(g), "acc": g["_ok"].mean()} for (l, e), g in df.groupby(GROUPS)]
    return pd.DataFrame(rows).sort_values("acc")


def mask(t):
    return "".join("L" if c.isalpha() else "#" if c.isdigit() else "*" if c == "*" else "." for c in t)


def visible(t):
    return [c for c in t.lower() if c.isalpha()]


def digits(t):
    return set(re.findall(r"\d", t))


def tok_sim(a, c):
    la, lc = len(a), len(c)
    if la == 0 or lc == 0:
        return 0.0
    len_score = 1.0 - abs(la - lc) / max(la, lc)
    va, vc = Counter(visible(a)), Counter(visible(c))
    inter = sum((va & vc).values())
    union = sum((va | vc).values())
    vis = inter / union if union else 0.0
    shape = SequenceMatcher(None, mask(a), mask(c)).ratio()
    pos = 0.0
    if la == lc:
        agree = sum(1 for x, y in zip(a.lower(), c.lower()) if x == y and x.isalpha())
        vp = sum(1 for x in a.lower() if x.isalpha())
        pos = agree / vp if vp else 0.0
    da, dc = digits(a), digits(c)
    dig = len(da & dc) / len(da | dc) if (da | dc) else 0.0
    return 0.34 * vis + 0.22 * len_score + 0.24 * shape + 0.12 * pos + 0.08 * dig


def tokens(s):
    return TOK.findall(str(s))


def bipartite_feats(anchor, cand):
    A, C = tokens(anchor), tokens(cand)
    if not A or not C:
        return {"bm_sum": 0, "bm_mean": 0, "bm_max": 0, "bm_min": 0, "bm_nstrong": 0, "bm_cov": 0, "bm_lw": 0, "na": len(A), "nc": len(C)}
    S = np.array([[tok_sim(a, c) for c in C] for a in A])
    ri, ci = linear_sum_assignment(-S)
    matched = S[ri, ci]
    lw = sum(matched[k] * len(A[ri[k]]) for k in range(len(ri))) / (sum(len(a) for a in A) + 1e-6)
    return {"bm_sum": float(matched.sum()), "bm_mean": float(matched.mean()), "bm_max": float(matched.max()),
            "bm_min": float(matched.min()), "bm_nstrong": int((matched > 0.5).sum()),
            "bm_cov": len(matched) / max(len(A), len(C)), "bm_lw": float(lw), "na": len(A), "nc": len(C)}


def global_feats(anchor, cand):
    a, c = str(anchor), str(cand)
    la, lc = len(a), len(c)
    va, vc = Counter(visible(a)), Counter(visible(c))
    inter = sum((va & vc).values())
    union = sum((va | vc).values())
    da, dc = digits(a), digits(c)
    return {"len_ratio": lc / (la + 1e-6), "len_absdiff": abs(la - lc),
            "vis_jac": inter / union if union else 0.0,
            "vis_overlap": inter / (min(sum(va.values()), sum(vc.values())) + 1e-6),
            "star_frac_a": a.count("*") / (la + 1e-6), "star_frac_c": c.count("*") / (lc + 1e-6),
            "dig_jac": len(da & dc) / len(da | dc) if (da | dc) else 0.0,
            "mask_ratio": SequenceMatcher(None, mask(a), mask(c)).ratio(),
            "seq_ratio": SequenceMatcher(None, a.lower(), c.lower()).ratio(),
            "ntok_ratio": len(tokens(c)) / (len(tokens(a)) + 1e-6)}


def build_structural(df):
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
            d["ctx_bm"] = bipartite_feats(loc, cand)["bm_mean"]
            per.append(d)
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
            rec = {"row": idx, "opt": oi, "target_language": r["target_language"], "entity_type": r["entity_type"]}
            rec.update(p)
            if "selected_option" in r:
                rec["y"] = int("ABCD"[oi] == r["selected_option"])
            recs.append(rec)
    return pd.DataFrame.from_records(recs)


def feature_cols(pair):
    excl = {"row", "opt", "target_language", "entity_type", "y", "fold"}
    return [c for c in pair.columns if c not in excl]


def to_matrix(pair, prob, n):
    m = np.zeros((n, 4))
    m[pair["row"].to_numpy(), pair["opt"].to_numpy()] = prob
    return m


def structural_gbm(train, test, folds):
    trp, tep = build_structural(train), build_structural(test)
    for d in (trp, tep):
        d["lang_code"] = d["target_language"].astype("category").cat.codes
        d["ent_code"] = d["entity_type"].astype("category").cat.codes
    fc = feature_cols(trp)
    trp["fold"] = folds[trp["row"].to_numpy()]
    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.02, num_leaves=64,
                  min_child_samples=50, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=3.0, verbose=-1, n_jobs=-1, seed=SEED)
    oof = np.zeros(len(trp))
    tp = np.zeros(len(tep))
    for f in range(N_SPLITS):
        tr, va = trp["fold"] != f, trp["fold"] == f
        m = lgb.train(params, lgb.Dataset(trp.loc[tr, fc], trp.loc[tr, "y"]), num_boost_round=4000,
                      valid_sets=[lgb.Dataset(trp.loc[va, fc], trp.loc[va, "y"])],
                      callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
        oof[va.to_numpy()] = m.predict(trp.loc[va, fc])
        tp += m.predict(tep[fc]) / N_SPLITS
    return to_matrix(trp, oof, len(train)), to_matrix(tep, tp, len(test))


def extra_redact(s, p, rng):
    return "".join("*" if (ch.isalpha() and rng.random() < p) else ch for ch in str(s))


def augment_df(df, p, seed):
    rng = random.Random(seed)
    d = df.copy()
    for c in CANDS:
        d[c] = df[c].map(lambda s: extra_redact(s, p, rng))
    def aug_src(s):
        m = ANCHOR.search(s)
        if not m:
            return s
        return s[:m.start()] + "[[ANCHOR]]" + extra_redact(m.group(1), p, rng) + "[[/ANCHOR]]" + s[m.end():]
    d["source_context"] = df["source_context"].map(aug_src)
    return d


def feats_tensor(pair, fc, n):
    X = np.zeros((n, 4, len(fc)), np.float32)
    X[pair["row"].to_numpy(), pair["opt"].to_numpy()] = pair[fc].to_numpy(np.float32)
    return X


def neural_matcher(train, test, folds):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    trp, tep = build_structural(train), build_structural(test)
    for d in (trp, tep):
        d["lang_code"] = d["target_language"].astype("category").cat.codes
        d["ent_code"] = d["entity_type"].astype("category").cat.codes
    fc = feature_cols(trp)
    aug = [build_structural(augment_df(train, 0.12 + 0.06 * k, 100 + k)) for k in range(2)]
    for a in aug:
        a["lang_code"] = a["target_language"].astype("category").cat.codes
        a["ent_code"] = a["entity_type"].astype("category").cat.codes
    y = y_index(train)
    Xtr = feats_tensor(trp, fc, len(train))
    Xte = feats_tensor(tep, fc, len(test))
    Xaug = [feats_tensor(a, fc, len(train)) for a in aug]
    mu = Xtr.reshape(-1, len(fc)).mean(0)
    sd = Xtr.reshape(-1, len(fc)).std(0) + 1e-6
    norm = lambda X: (X - mu) / sd
    Xtr, Xte = norm(Xtr), norm(Xte)
    Xaug = [norm(a) for a in Xaug]

    class MLP(nn.Module):
        def __init__(self, F, h=256, p=0.3):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(F), nn.Linear(F, h), nn.GELU(), nn.Dropout(p),
                                     nn.Linear(h, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))
        def forward(self, x):
            return self.net(x).squeeze(-1)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    oof = np.zeros((len(train), 4))
    tp = np.zeros((len(test), 4))
    for f in range(N_SPLITS):
        tri, vai = np.where(folds != f)[0], np.where(folds == f)[0]
        Xt = np.concatenate([Xtr[tri]] + [a[tri] for a in Xaug], 0)
        yt = np.concatenate([y[tri]] * (1 + len(Xaug)), 0)
        Xt = torch.tensor(Xt, device=dev)
        yt = torch.tensor(yt, device=dev)
        Xv = torch.tensor(Xtr[vai], device=dev)
        yv = y[vai]
        Xtest = torch.tensor(Xte, device=dev)
        model = MLP(len(fc)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        best, bstate = -1, None
        for ep in range(40):
            model.train()
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                lossf(model(Xt[idx]), yt[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1).cpu().numpy() == yv).mean()
            if acc > best:
                best = acc
                bstate = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(bstate)
        model.eval()
        with torch.no_grad():
            oof[vai] = model(Xv).softmax(-1).cpu().numpy()
            tp += model(Xtest).softmax(-1).cpu().numpy() / N_SPLITS
    return oof, tp


def load_transformer(tag):
    o, t = OOF / f"oof_{tag}.npy", OOF / f"test_{tag}.npy"
    if o.exists() and t.exists():
        return np.load(o), np.load(t)
    return None, None


def normalize(p):
    p = np.clip(p, 1e-6, None)
    return p / p.sum(1, keepdims=True)


def optimize_weights(train, oofs, iters=30000):
    rng = np.random.default_rng(0)
    n = len(oofs)
    stack = np.stack([normalize(o) for o in oofs])
    y = y_index(train)
    grp = (train["target_language"].astype(str) + "|" + train["entity_type"].astype(str)).astype("category").cat.codes.to_numpy()
    ng = grp.max() + 1
    counts = np.bincount(grp, minlength=ng).astype(float)
    def score(w):
        pred = (stack * w[:, None, None]).sum(0).argmax(1)
        ok = (pred == y).astype(float)
        return float((np.bincount(grp, weights=ok, minlength=ng) / np.maximum(counts, 1)).mean())
    best_w = np.ones(n) / n
    best = score(best_w)
    for _ in range(iters):
        w = rng.dirichlet(np.ones(n) * 0.7)
        s = score(w)
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
                s = score(w)
                if s > best:
                    best, best_w = s, w
    return best_w / best_w.sum(), best


def main():
    train, test = load_data()
    folds = add_folds(train)
    models = {}
    print("training structural GBM ...")
    models["struct"] = structural_gbm(train, test, folds)
    print("training neural matcher ...")
    models["nn"] = neural_matcher(train, test, folds)
    for tag in TRANSFORMER_TAGS:
        o, t = load_transformer(tag)
        if o is not None:
            models[tag] = (o, t)
            print(f"loaded transformer {tag}")
    scores = {t: balanced_macro(train, normalize(models[t][0]).argmax(1)) for t in models}
    for t in models:
        print(f"  {t:10s} OOF balanced macro {scores[t]:.4f}")
    tags = [t for t in models if scores[t] >= 0.52]
    oofs = [normalize(models[t][0]) for t in tags]
    tests = [normalize(models[t][1]) for t in tags]
    print("ensemble members (equal weight):", tags)
    blend = np.mean(np.stack(tests), 0)
    blend_oof = np.mean(np.stack(oofs), 0)
    print("ensemble OOF balanced macro", round(balanced_macro(train, blend_oof.argmax(1)), 4))
    print(per_group(train, blend_oof.argmax(1)).to_string(index=False))
    sub = pd.DataFrame({"id": test["id"], "selected_option": [LETTERS[i] for i in blend.argmax(1)]})
    sub.to_csv(SUB / "submission.csv", index=False)
    print("wrote", SUB / "submission.csv", len(sub), "rows")


if __name__ == "__main__":
    main()
