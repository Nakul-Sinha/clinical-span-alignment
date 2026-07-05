"""Data loading, anchor extraction, redaction-aware helpers, and CV folds."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CANDS = ["candidate_a", "candidate_b", "candidate_c", "candidate_d"]
LETTERS = ["A", "B", "C", "D"]
LANGS = ["cz", "en", "it", "nl", "ro", "sv"]
ENTITIES = ["disease", "procedure", "symptom"]

_ANCHOR = re.compile(r"\[\[ANCHOR\]\](.*?)\[\[/ANCHOR\]\]", re.DOTALL)


def default_data_dir() -> Path:
    return Path(r"G:\Datacurve\clinical chal\dataset")


def extract_anchor(source_context: str) -> str:
    m = _ANCHOR.search(source_context)
    return m.group(1).strip() if m else ""


def anchor_span(source_context: str):
    """Return (anchor_text, left_context, right_context) around the anchor."""
    m = _ANCHOR.search(source_context)
    if not m:
        return "", source_context, ""
    return m.group(1).strip(), source_context[:m.start()], source_context[m.end():]


def load(data_dir: Path | str | None = None):
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    for df in (train, test):
        df["anchor"] = df["source_context"].map(extract_anchor)
    return train, test


def add_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42,
              fold_col: str = "fold") -> pd.DataFrame:
    """Stratified folds on the (language, entity) group so every fold covers all 18 groups."""
    df = df.copy()
    strat = df["target_language"].astype(str) + "|" + df["entity_type"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df[fold_col] = -1
    for i, (_, va) in enumerate(skf.split(df, strat)):
        df.iloc[va, df.columns.get_loc(fold_col)] = i
    return df


def label_to_idx(df: pd.DataFrame) -> np.ndarray:
    return df["selected_option"].map({l: i for i, l in enumerate(LETTERS)}).to_numpy()
