"""EDA for Cross-Lingual Clinical Span Anchoring."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"G:\ml\clinical chal\dataset")

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sub = pd.read_csv(DATA / "sample_submission.csv")

print("=== shapes ===")
print("train:", train.shape, "test:", test.shape, "sub:", sub.shape)
print("\ntrain columns:", list(train.columns))
print("test columns:", list(test.columns))

print("\n=== nulls ===")
print(train.isna().sum().to_dict())
print(test.isna().sum().to_dict())

print("\n=== group counts train (lang x entity) ===")
print(train.groupby(["target_language", "entity_type"]).size().unstack(fill_value=0))
print("\n=== group counts test ===")
print(test.groupby(["target_language", "entity_type"]).size().unstack(fill_value=0))

print("\n=== selected_option distribution ===")
print(train["selected_option"].value_counts(normalize=True).sort_index())
print("\nper language:")
print(train.groupby("target_language")["selected_option"].value_counts(normalize=True).unstack().round(3))

ANCH = re.compile(r"\[\[ANCHOR\]\](.*?)\[\[/ANCHOR\]\]", re.DOTALL)

def get_anchor(s):
    m = ANCH.search(s)
    return m.group(1) if m else ""

train["anchor"] = train["source_context"].map(get_anchor)
test["anchor"] = test["source_context"].map(get_anchor)
print("\n=== anchor extraction ===")
print("train rows with anchor:", (train["anchor"] != "").mean())
print("test rows with anchor:", (test["anchor"] != "").mean())
print("anchor char len stats:", train["anchor"].str.len().describe().round(1).to_dict())

cands = ["candidate_a", "candidate_b", "candidate_c", "candidate_d"]

print("\n=== candidate properties (train) ===")
for c in cands:
    in_ctx = train.apply(lambda r: str(r[c]) in str(r["target_context"]), axis=1)
    print(f"{c}: appears verbatim in target_context: {in_ctx.mean():.4f}, len stats: mean={train[c].str.len().mean():.1f} max={train[c].str.len().max()}")

# occurrences count of each candidate in context
def occ_count(r, c):
    return str(r["target_context"]).count(str(r[c]))

occ = train.apply(lambda r: [occ_count(r, c) for c in cands], axis=1)
occ = np.array(occ.tolist())
print("occurrence counts distribution (all candidates pooled):")
vals, cnts = np.unique(occ, return_counts=True)
print(dict(zip(vals.tolist(), cnts.tolist())))

# duplicates among candidates within a row
dup = train.apply(lambda r: len({r[c] for c in cands}) < 4, axis=1)
print("rows with duplicate candidate strings:", dup.mean())

print("\n=== context length stats (chars) ===")
for col in ["source_context", "target_context"]:
    print(col, "train:", train[col].str.len().describe().round(0).to_dict())

print("\n=== sample rows ===")
pd.set_option("display.width", 250)
for i in [0, 1, 2]:
    r = train.iloc[i]
    print("-" * 100)
    print("id:", r["id"], "| lang:", r["target_language"], "| type:", r["entity_type"], "| ans:", r["selected_option"])
    print("SOURCE:", r["source_context"][:600])
    print("TARGET:", r["target_context"][:600])
    for c in cands:
        print(f"  {c[-1].upper()}: {r[c]}")

# ---- quick signal probe: char n-gram similarity anchor vs candidates ----
def ngrams(s, n=3):
    s = f" {s.lower()} "
    return {s[i:i+n] for i in range(max(1, len(s)-n+1))}

def sim(a, b):
    A, B = ngrams(a), ngrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def macro_balanced(df, pred_col):
    scores = []
    for _, g in df.groupby(["target_language", "entity_type"]):
        scores.append((g[pred_col] == g["selected_option"]).mean())
    return float(np.mean(scores))

letters = np.array(["A", "B", "C", "D"])
sims = train.apply(lambda r: [sim(r["anchor"], str(r[c])) for c in cands], axis=1)
sims = np.array(sims.tolist())
train["pred_sim"] = letters[sims.argmax(axis=1)]
print("\n=== heuristic: max char-3gram jaccard(anchor, candidate) ===")
print("plain accuracy:", (train["pred_sim"] == train["selected_option"]).mean().round(4))
print("balanced macro accuracy:", round(macro_balanced(train, "pred_sim"), 4))
per_grp = train.groupby(["target_language", "entity_type"]).apply(
    lambda g: (g["pred_sim"] == g["selected_option"]).mean(), include_groups=False).unstack().round(3)
print(per_grp)

# jsonl peek
with open(DATA / "train.jsonl", encoding="utf-8") as f:
    line = json.loads(f.readline())
print("\n=== train.jsonl first record keys ===")
print(list(line.keys()))
print(json.dumps(line, ensure_ascii=False)[:1200])
