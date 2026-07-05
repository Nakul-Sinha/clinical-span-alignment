"""Pairwise (anchor, candidate) features for the gradient-boosting baseline.

The redaction is aggressive (~30% of chars are '*'), candidates are re-redacted
differently from the context, and 21% of anchor/candidate pairs share no char
3-gram. So features focus on the few signals that survive: fuzzy morpheme overlap,
shared numbers, length/shape priors, and *relative* rank across the four options.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from .data import CANDS, anchor_span

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z]+\d?|\*+")


def char_ngrams(s: str, n: int) -> set:
    s = f" {s.strip().lower()} "
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coef(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def numbers(s: str) -> set:
    return set(_NUM.findall(s))


def star_shape(s: str) -> str:
    """Reduce a token to its star/letter/digit skeleton, e.g. 'bil****al' -> 'LLL****LL'."""
    out = []
    for ch in s:
        if ch == "*":
            out.append("*")
        elif ch.isdigit():
            out.append("#")
        elif ch.isalpha():
            out.append("L")
        else:
            out.append(ch)
    return "".join(out)


def _pair_feats(anchor: str, cand: str) -> dict:
    a = str(anchor)
    c = str(cand)
    a2, a3, a4 = char_ngrams(a, 2), char_ngrams(a, 3), char_ngrams(a, 4)
    c2, c3, c4 = char_ngrams(c, 2), char_ngrams(c, 3), char_ngrams(c, 4)
    an, cn = numbers(a), numbers(c)
    la, lc = len(a), len(c)
    astars, cstars = a.count("*"), c.count("*")
    f = {
        "jac2": jaccard(a2, c2),
        "jac3": jaccard(a3, c3),
        "jac4": jaccard(a4, c4),
        "ov2": overlap_coef(a2, c2),
        "ov3": overlap_coef(a3, c3),
        "seqratio": SequenceMatcher(None, a.lower(), c.lower()).ratio(),
        "num_shared": len(an & cn),
        "num_jac": jaccard(an, cn),
        "cand_has_num": float(len(cn) > 0),
        "anchor_has_num": float(len(an) > 0),
        "len_cand": lc,
        "len_anchor": la,
        "len_ratio": lc / (la + 1e-6),
        "len_absdiff": abs(lc - la),
        "cand_star_frac": cstars / (lc + 1e-6),
        "anchor_star_frac": astars / (la + 1e-6),
        "star_frac_diff": abs(cstars / (lc + 1e-6) - astars / (la + 1e-6)),
        "cand_ntok": len(c.split()),
        "anchor_ntok": len(a.split()),
        "first3_match": float(a[:3].lower() == c[:3].lower()),
        "last3_match": float(a[-3:].lower() == c[-3:].lower()),
    }
    return f


def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (example, option). Includes absolute + within-row relative features."""
    records = []
    base_feat_keys = None
    for idx, r in df.iterrows():
        anchor, left, right = anchor_span(r["source_context"])
        # local Spanish context around the anchor (last/first tokens) — weak cross-lingual cue
        left_ctx = " ".join(left.split()[-6:])
        right_ctx = " ".join(right.split()[:6])
        loc = f"{left_ctx} {right_ctx}"
        loc3 = char_ngrams(loc, 3)
        per_opt = []
        for oi, col in enumerate(CANDS):
            cand = str(r[col])
            feats = _pair_feats(anchor, cand)
            feats["ctx_jac3"] = jaccard(loc3, char_ngrams(cand, 3))
            per_opt.append(feats)
        if base_feat_keys is None:
            base_feat_keys = list(per_opt[0].keys())
        # within-row relative features on the main similarity signals
        for key in ["jac3", "jac2", "seqratio", "ov3", "num_shared", "len_ratio"]:
            vals = np.array([po[key] for po in per_opt], dtype=float)
            order = vals.argsort().argsort()  # 0..3 rank ascending
            mx = vals.max()
            for oi, po in enumerate(per_opt):
                po[f"{key}_rank"] = float(order[oi])
                po[f"{key}_is_max"] = float(vals[oi] == mx and mx > 0)
                # margin to the best *other* option
                others = np.delete(vals, oi)
                po[f"{key}_margin"] = float(vals[oi] - others.max())
        for oi, (col, po) in enumerate(zip(CANDS, per_opt)):
            rec = {"row": idx, "id": r["id"], "opt": oi,
                   "target_language": r["target_language"],
                   "entity_type": r["entity_type"]}
            rec.update(po)
            if "selected_option" in r:
                rec["y"] = int("ABCD"[oi] == r["selected_option"])
            records.append(rec)
    out = pd.DataFrame.from_records(records)
    return out


FEATURE_COLS_EXCLUDE = {"row", "id", "opt", "target_language", "entity_type", "y"}


def feature_columns(pair_df: pd.DataFrame) -> list:
    return [c for c in pair_df.columns if c not in FEATURE_COLS_EXCLUDE]
