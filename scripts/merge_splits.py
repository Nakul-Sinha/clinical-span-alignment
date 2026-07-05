"""Merge fold-split kernel outputs (e.g. xlmr-large-a folds 0-2, -b folds 3-4)
into a single OOF/test pair, then remove the split copies from oof/ so the
ensemble sees one model.

Usage: python scripts/merge_splits.py <out_tag> <tagA> <nA> <tagB> <nB> ...
"""
import sys
from pathlib import Path
import numpy as np

OOF = Path(__file__).resolve().parents[1] / "oof"


def main():
    out_tag = sys.argv[1]
    rest = sys.argv[2:]
    pairs = [(rest[i], int(rest[i + 1])) for i in range(0, len(rest), 2)]
    oof_sum = None
    test_num = None
    denom = 0
    for tag, n in pairs:
        o = np.load(OOF / f"oof_{tag}.npy")
        t = np.load(OOF / f"test_{tag}.npy")
        oof_sum = o if oof_sum is None else oof_sum + o
        test_num = n * t if test_num is None else test_num + n * t
        denom += n
    test = test_num / denom
    # sanity: OOF rows should be covered exactly once (no overlap -> at most one nonzero group per row)
    covered = (oof_sum.sum(1) > 0).mean()
    print(f"merged {out_tag}: OOF coverage={covered:.3f} (expect ~1.0), folds={denom}")
    np.save(OOF / f"oof_{out_tag}.npy", oof_sum)
    np.save(OOF / f"test_{out_tag}.npy", test)
    for tag, _ in pairs:
        for pre in ("oof", "test"):
            p = OOF / f"{pre}_{tag}.npy"
            if p.exists():
                p.unlink()
    print("wrote", f"oof_{out_tag}.npy", f"test_{out_tag}.npy")


if __name__ == "__main__":
    main()
