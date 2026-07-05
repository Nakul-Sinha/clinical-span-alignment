"""Neural structural matcher: an MLP with listwise softmax over the 4 options on the
rich structural features, trained with REDACTION AUGMENTATION (extra random star-
masking of anchor/candidates, length-preserving) for robustness + capacity beyond
the GBM. Diverse ensemble member. Saves oof_nn / test_nn.
"""
import random, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clinspan.data import load, add_folds, CANDS, LETTERS
from clinspan.metric import balanced_macro_accuracy, per_group_accuracy
from structural_gbm import build  # reuse the structural feature builder

OOF = ROOT / "oof"
N_SPLITS, SEED = 5, 42
_ANCHOR = re.compile(r"\[\[ANCHOR\]\](.*?)\[\[/ANCHOR\]\]", re.DOTALL)
device = "cuda" if torch.cuda.is_available() else "cpu"


def extra_redact(s, p, rng):
    return "".join("*" if (ch.isalpha() and rng.random() < p) else ch for ch in str(s))


def augment_df(df, p, seed):
    rng = random.Random(seed)
    d = df.copy()
    for c in CANDS:
        d[c] = df[c].map(lambda s: extra_redact(s, p, rng))
    def aug_src(s):
        m = _ANCHOR.search(s)
        if not m:
            return s
        return s[:m.start()] + "[[ANCHOR]]" + extra_redact(m.group(1), p, rng) + "[[/ANCHOR]]" + s[m.end():]
    d["source_context"] = df["source_context"].map(aug_src)
    return d


def feats_matrix(pair, feat_cols, n_rows):
    """(n_rows, 4, F) feature tensor + label per row."""
    F = len(feat_cols)
    X = np.zeros((n_rows, 4, F), np.float32)
    X[pair["row"].to_numpy(), pair["opt"].to_numpy()] = pair[feat_cols].to_numpy(np.float32)
    return X


class MLP(nn.Module):
    def __init__(self, F, h=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(F), nn.Linear(F, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, 1))

    def forward(self, x):           # x: (B,4,F)
        return self.net(x).squeeze(-1)   # (B,4)


def main():
    train, test = load()
    train = add_folds(train, n_splits=N_SPLITS, seed=SEED)
    print("building base structural features...")
    trp = build(train); tep = build(test)
    excl = {"row", "opt", "target_language", "entity_type", "y", "fold"}
    feat_cols = [c for c in trp.columns if c not in excl]

    # redaction-augmented feature copies (train only)
    aug_frames = []
    for k in range(2):
        print(f"building redaction-augmented features (pass {k})...")
        aug = build(augment_df(train, p=0.12 + 0.06 * k, seed=100 + k))
        aug_frames.append(aug)

    y = train["selected_option"].map({l: i for i, l in enumerate(LETTERS)}).to_numpy()
    folds = train["fold"].to_numpy()
    Xtr = feats_matrix(trp, feat_cols, len(train))
    Xte = feats_matrix(tep, feat_cols, len(test))
    Xaug = [feats_matrix(a, feat_cols, len(train)) for a in aug_frames]

    # standardize with train stats
    mu = Xtr.reshape(-1, len(feat_cols)).mean(0); sd = Xtr.reshape(-1, len(feat_cols)).std(0) + 1e-6
    norm = lambda X: (X - mu) / sd
    Xtr_n, Xte_n = norm(Xtr), norm(Xte)
    Xaug_n = [norm(a) for a in Xaug]

    torch.manual_seed(SEED); np.random.seed(SEED)
    oof = np.zeros((len(train), 4)); test_p = np.zeros((len(test), 4))
    for fold in range(N_SPLITS):
        tri = np.where(folds != fold)[0]; vai = np.where(folds == fold)[0]
        # stack original + augmented training rows
        Xt = np.concatenate([Xtr_n[tri]] + [a[tri] for a in Xaug_n], 0)
        yt = np.concatenate([y[tri]] * (1 + len(Xaug_n)), 0)
        Xt = torch.tensor(Xt, device=device); yt = torch.tensor(yt, device=device)
        Xv = torch.tensor(Xtr_n[vai], device=device); yv = y[vai]
        Xtest = torch.tensor(Xte_n, device=device)

        model = MLP(len(feat_cols)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        best_acc, best_state = -1, None
        bs = 256
        for epoch in range(40):
            model.train(); perm = torch.randperm(len(Xt), device=device)
            for i in range(0, len(Xt), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1).cpu().numpy() == yv).mean()
            if acc > best_acc:
                best_acc = acc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof[vai] = model(Xv).softmax(-1).cpu().numpy()
            test_p += model(Xtest).softmax(-1).cpu().numpy() / N_SPLITS
        print(f"  fold {fold}: best_val_acc={best_acc:.4f}")

    train["prediction"] = [LETTERS[i] for i in oof.argmax(1)]
    print("\n=== neural matcher OOF balanced macro:", round(balanced_macro_accuracy(train), 4), "===")
    print(per_group_accuracy(train).to_string(index=False))
    np.save(OOF / "oof_nn.npy", oof); np.save(OOF / "test_nn.npy", test_p)
    print("saved oof_nn / test_nn")


if __name__ == "__main__":
    main()
