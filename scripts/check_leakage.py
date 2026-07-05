"""Check for note-level duplication that could leak across CV folds / train-test."""
import re
from pathlib import Path
import pandas as pd

DATA = Path(r"G:\ml\clinical chal\dataset")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

for name, df in [("train", train), ("test", test)]:
    print(f"=== {name} ({len(df)} rows) ===")
    print("  unique source_context:", df["source_context"].nunique())
    print("  unique target_context:", df["target_context"].nunique())
    print("  unique id:", df["id"].nunique())

# overlap of contexts between train and test (potential leakage / shared notes)
print("\n=== train/test context overlap ===")
print("  shared source_context:", len(set(train["source_context"]) & set(test["source_context"])))
print("  shared target_context:", len(set(train["target_context"]) & set(test["target_context"])))

# how many rows share a source_context (same Spanish note, multiple anchors)?
vc = train["source_context"].value_counts()
print("\nsource_context multiplicity in train:", vc.value_counts().sort_index().to_dict())
vt = train["target_context"].value_counts()
print("target_context multiplicity in train:", vt.value_counts().sort_index().to_dict())

# Build a note key by stripping the anchor markers from source, to group true notes
def strip_anchor(s):
    return re.sub(r"\[\[/?ANCHOR\]\]", "", s)

train["src_note"] = train["source_context"].map(strip_anchor)
print("\nunique src_note (anchor stripped):", train["src_note"].nunique(),
      "-> avg rows/note:", round(len(train) / train["src_note"].nunique(), 2))
# combined note key
train["note_key"] = train["src_note"] + "|||" + train["target_context"]
print("unique (src_note,target) pairs:", train["note_key"].nunique())
