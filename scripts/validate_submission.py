"""Validate a submission against the challenge requirements.

- exactly len(test) rows + header
- columns exactly [id, selected_option]
- every test id present exactly once
- selected_option in {A, B, C, D}
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clinspan.data import load  # noqa


def validate(path):
    _, test = load()
    sub = pd.read_csv(path, dtype=str)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK  " if cond else " FAIL ") + msg)
        ok = ok and cond

    check(list(sub.columns) == ["id", "selected_option"], f"columns == [id, selected_option] (got {list(sub.columns)})")
    check(len(sub) == len(test), f"row count == {len(test)} (got {len(sub)})")
    check(sub["id"].nunique() == len(sub), "ids unique")
    check(set(sub["id"]) == set(test["id"]), "id set matches test.csv exactly")
    check(sub["selected_option"].isin(list("ABCD")).all(), "selected_option in {A,B,C,D}")
    check(sub["selected_option"].notna().all(), "no missing predictions")
    print("distribution:", sub["selected_option"].value_counts().sort_index().to_dict())
    print("VALID" if ok else "INVALID")
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "submissions" / "submission.csv")
    sys.exit(0 if validate(path) else 1)
