"""Probe the redaction scheme and candidate/context alignment more precisely."""
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(r"G:\Datacurve\clinical chal\dataset")
train = pd.read_csv(DATA / "train.csv")
ANCH = re.compile(r"\[\[ANCHOR\]\](.*?)\[\[/ANCHOR\]\]", re.DOTALL)
cands = ["candidate_a", "candidate_b", "candidate_c", "candidate_d"]
train["anchor"] = train["source_context"].map(lambda s: (ANCH.search(s).group(1) if ANCH.search(s) else "").strip())

# character inventory
from collections import Counter
allchars = Counter("".join(train["target_context"].sample(500, random_state=0)))
print("Top target_context chars:", allchars.most_common(25))

# token shapes: classify tokens
def classify(tok):
    if re.fullmatch(r"[a-z]+\d", tok):
        return "skeleton_digit"      # e.g. trt9
    if "*" in tok:
        return "star_mask"           # e.g. bil****al
    if re.fullmatch(r"\d+([.,]\d+)*", tok):
        return "number"
    if re.fullmatch(r"[a-z]+", tok):
        return "plain_word"
    return "other"

toks = re.findall(r"\S+", " ".join(train["target_context"].sample(300, random_state=1)))
toks = [re.sub(r"^[^\w*]+|[^\w*]+$", "", t) for t in toks]
toks = [t for t in toks if t]
c = Counter(classify(t) for t in toks)
print("\nToken shape distribution (target_context sample):", c)

# does the correct candidate have a fuzzy near-duplicate in target_context?
def ngrams(s, n=3):
    s = f" {s} "
    return [s[i:i+n] for i in range(max(1, len(s)-n+1))]

def best_window_sim(cand, ctx, n=3):
    """max jaccard between candidate ngrams and any equal-length char window of ctx."""
    cg = set(ngrams(cand, n))
    if not cg:
        return 0.0
    L = len(cand)
    best = 0.0
    step = max(1, L // 4)
    for i in range(0, max(1, len(ctx) - L + 1), step):
        w = ctx[i:i+L]
        wg = set(ngrams(w, n))
        if wg:
            j = len(cg & wg) / len(cg | wg)
            if j > best:
                best = j
    return best

sample = train.sample(200, random_state=2)
corr_sim, wrong_sim = [], []
for _, r in sample.iterrows():
    correct = r[f"candidate_{r['selected_option'].lower()}"]
    for c_ in cands:
        s = best_window_sim(str(r[c_]), str(r["target_context"]))
        if r[c_] == correct:
            corr_sim.append(s)
        else:
            wrong_sim.append(s)
print(f"\nBest-window sim of candidate vs target_context:")
print(f"  correct candidate: mean={np.mean(corr_sim):.3f}")
print(f"  wrong candidates:  mean={np.mean(wrong_sim):.3f}")

# anchor vs candidate similarity separation (the core cross-lingual signal)
def jac(a, b, n=3):
    A, B = set(ngrams(a, n)), set(ngrams(b, n))
    return len(A & B) / len(A | B) if A and B else 0.0

corr, wrong = [], []
for _, r in train.iterrows():
    correct = r[f"candidate_{r['selected_option'].lower()}"]
    for c_ in cands:
        s = jac(str(r["anchor"]), str(r[c_]))
        (corr if r[c_] == correct else wrong).append(s)
print(f"\nAnchor vs candidate char-3gram jaccard separation (full train):")
print(f"  correct: mean={np.mean(corr):.3f} median={np.median(corr):.3f}")
print(f"  wrong:   mean={np.mean(wrong):.3f} median={np.median(wrong):.3f}")

# how often is correct the argmax, and how often tied at 0?
def row_pred(r, n=3):
    sims = [jac(str(r["anchor"]), str(r[c_]), n) for c_ in cands]
    return sims
allzero = 0
for _, r in train.iterrows():
    sims = row_pred(r)
    if max(sims) == 0:
        allzero += 1
print(f"\nRows where all anchor-candidate sims are 0 (no 3gram overlap): {allzero} ({allzero/len(train):.3f})")
