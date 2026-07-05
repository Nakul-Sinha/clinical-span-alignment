"""Balanced macro accuracy — the official challenge metric."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_COLS = ["target_language", "entity_type"]


def balanced_macro_accuracy(df: pd.DataFrame, pred_col: str = "prediction",
                            label_col: str = "selected_option") -> float:
    """Accuracy per (target_language, entity_type) group, averaged over groups."""
    scores = []
    for _, g in df.groupby(GROUP_COLS):
        scores.append((g[pred_col] == g[label_col]).mean())
    return float(np.mean(scores))


def per_group_accuracy(df: pd.DataFrame, pred_col: str = "prediction",
                       label_col: str = "selected_option") -> pd.DataFrame:
    rows = []
    for (lang, ent), g in df.groupby(GROUP_COLS):
        rows.append({"target_language": lang, "entity_type": ent,
                     "n": len(g), "acc": (g[pred_col] == g[label_col]).mean()})
    out = pd.DataFrame(rows).sort_values("acc")
    return out
