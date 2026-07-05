"""Contrastive bi-encoder for cross-lingual clinical span anchoring.

Two-tower (shared multilingual encoder): encode the Spanish source+anchor query and
each redacted target candidate SEPARATELY, mean-pool, L2-normalize, score by scaled
cosine similarity, and train listwise (softmax over the 4 candidates, the 3 distractors
as in-example contrastive negatives). Different inductive bias from the cross-encoders
-> a diverse ensemble member and a semantic lever for hard (low-resource) groups.

Self-contained Kaggle kernel. Config injected by kaggle/push.py.
"""
import hashlib, glob, json, os, time
import numpy as np
import pandas as pd

# === CONFIG_JSON_START ===
CONFIG = {
    "model_name": "xlm-roberta-base", "tag": "bienc", "q_len": 160, "c_len": 48,
    "epochs": 4, "lr": 2e-5, "batch_size": 16, "grad_accum": 1, "warmup_ratio": 0.1,
    "weight_decay": 0.01, "n_splits": 5, "seed": 42, "folds_to_run": [0, 1, 2, 3, 4],
    "max_grad_norm": 1.0, "scale": 20.0, "head_warmup_steps": 40, "max_attempts": 3,
    "collapse_thresh": 0.32, "subset": 0,
}
# === CONFIG_JSON_END ===

OUT = "/kaggle/working"
CANDS = ["candidate_a", "candidate_b", "candidate_c", "candidate_d"]
LETTERS = ["A", "B", "C", "D"]
ANCHOR_TOKENS = ["[[ANCHOR]]", "[[/ANCHOR]]"]


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def det_folds(df, n, seed):
    h = lambda s: int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16)
    df = df.copy(); df["_h"] = df["id"].map(h); df["fold"] = -1
    for _, idx in df.groupby(["target_language", "entity_type"]).groups.items():
        sub = df.loc[idx].sort_values("_h"); df.loc[sub.index, "fold"] = np.arange(len(sub)) % n
    return df["fold"].to_numpy()


def find_csv(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return sorted(hits, key=len)[0]


def main():
    import torch, torch.nn as nn, torch.nn.functional as Fn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    cfg = CONFIG; dev = "cuda"
    log("torch", torch.__version__, "| config", json.dumps(cfg))

    def seed_all(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    seed_all(cfg["seed"])
    train = pd.read_csv(find_csv("train.csv")); test = pd.read_csv(find_csv("test.csv"))
    if cfg["subset"]:
        train = train.groupby(["target_language", "entity_type"], group_keys=False).head(6).reset_index(drop=True)
        test = test.head(cfg["subset"]).reset_index(drop=True)
    folds = det_folds(train, cfg["n_splits"], cfg["seed"])
    y = train["selected_option"].map({l: i for i, l in enumerate(LETTERS)}).to_numpy()
    for df in (train, test):
        df["_q"] = "language: " + df["target_language"] + " | entity: " + df["entity_type"] + " | " + df["source_context"]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.add_special_tokens({"additional_special_tokens": ANCHOR_TOKENS})

    class DS(Dataset):
        def __init__(self, df, lab=None):
            self.q = df["_q"].tolist(); self.c = [[str(df.iloc[i][k]) for k in CANDS] for i in range(len(df))]
            self.y = lab if lab is not None else [-1] * len(df)
        def __len__(self): return len(self.q)
        def __getitem__(self, i): return self.q[i], self.c[i], int(self.y[i])

    def collate(b):
        qs, cs, ys = zip(*b)
        q = tok(list(qs), truncation=True, max_length=cfg["q_len"], padding=True, return_tensors="pt")
        flat = [c for cc in cs for c in cc]
        c = tok(flat, truncation=True, max_length=cfg["c_len"], padding=True, return_tensors="pt")
        B = len(qs)
        c = {k: v.view(B, 4, -1) for k, v in c.items()}
        return q, c, torch.tensor(ys)

    class BiEnc(nn.Module):
        def __init__(self, name, vocab):
            super().__init__(); self.bb = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
            self.bb.resize_token_embeddings(vocab); self.scale = cfg["scale"]
        def enc(self, ids, am):
            h = self.bb(input_ids=ids, attention_mask=am).last_hidden_state
            m = am.unsqueeze(-1).to(h.dtype); v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            return Fn.normalize(v, dim=-1)
        def forward(self, q, c):
            qv = self.enc(q["input_ids"], q["attention_mask"])              # (B,H)
            B, C, L = c["input_ids"].shape
            cv = self.enc(c["input_ids"].view(B * C, L), c["attention_mask"].view(B * C, L)).view(B, C, -1)
            return (qv.unsqueeze(1) * cv).sum(-1) * self.scale             # (B,4)

    @torch.no_grad()
    def predict(model, df):
        model.eval(); dl = DataLoader(DS(df), batch_size=cfg["batch_size"] * 2, collate_fn=collate)
        out = []
        for q, c, _ in dl:
            q = {k: v.to(dev) for k, v in q.items()}; c = {k: v.to(dev) for k, v in c.items()}
            with torch.amp.autocast("cuda"): out.append(model(q, c).float().softmax(-1).cpu().numpy())
        return np.concatenate(out, 0)

    def run_fold(fold, seed):
        seed_all(seed)
        tr, va = train[folds != fold].reset_index(drop=True), train[folds == fold].reset_index(drop=True)
        vy = y[folds == fold]
        model = BiEnc(cfg["model_name"], len(tok)).to(dev)
        nd = ["bias", "LayerNorm.weight"]
        params = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in nd)], "weight_decay": cfg["weight_decay"]},
                  {"params": [p for n, p in model.named_parameters() if any(x in n for x in nd)], "weight_decay": 0.0}]
        opt = torch.optim.AdamW(params, lr=cfg["lr"])
        dl = DataLoader(DS(tr, y[folds != fold]), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, drop_last=True)
        tot = (len(dl) // cfg["grad_accum"]) * cfg["epochs"]
        sch = get_linear_schedule_with_warmup(opt, int(tot * cfg["warmup_ratio"]), tot)
        scaler = torch.amp.GradScaler("cuda"); lossf = nn.CrossEntropyLoss()
        hw = cfg["head_warmup_steps"]
        if hw > 0: model.bb.requires_grad_(False)
        g = 0; best, bstate = -1.0, None
        for ep in range(cfg["epochs"]):
            model.train(); opt.zero_grad(); t0 = time.time()
            for bi, (q, c, lb) in enumerate(dl):
                q = {k: v.to(dev) for k, v in q.items()}; c = {k: v.to(dev) for k, v in c.items()}; lb = lb.to(dev)
                with torch.amp.autocast("cuda"): loss = lossf(model(q, c), lb) / cfg["grad_accum"]
                scaler.scale(loss).backward()
                if (bi + 1) % cfg["grad_accum"] == 0:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(); g += 1
                    if hw > 0 and g == hw: model.bb.requires_grad_(True)
            acc = (predict(model, va).argmax(1) == vy).mean()
            log(f"    ep{ep} val_acc={acc:.4f} ({time.time()-t0:.0f}s)")
            if acc > best: best = acc; bstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(bstate)
        of, tp = predict(model, va), predict(model, test)
        del model; torch.cuda.empty_cache()
        return float(best), of, tp

    oof = np.zeros((len(train), 4)); tp = np.zeros((len(test), 4))
    for fold in cfg["folds_to_run"]:
        log(f"===== fold {fold} =====")
        ba, of_, tp_ = -1, None, None
        for att in range(cfg["max_attempts"]):
            b, o, t = run_fold(fold, cfg["seed"] + fold + att * 1000)
            if b > ba: ba, of_, tp_ = b, o, t
            if b >= cfg["collapse_thresh"]: break
            log(f"  retry (val={b:.3f})")
        oof[folds == fold] = of_; tp += tp_ / len(cfg["folds_to_run"])
        log(f"  fold {fold} best={ba:.4f}")

    ran = np.isin(folds, cfg["folds_to_run"]); pred = oof.argmax(1)
    train["_c"] = (pred == y)
    macro = np.mean([train[ran & (train.target_language == l) & (train.entity_type == e)]["_c"].mean()
                     for l in train.target_language.unique() for e in train.entity_type.unique()
                     if ((train.target_language == l) & (train.entity_type == e) & ran).any()])
    log(f"bi-encoder OOF balanced macro: {macro:.4f}")
    np.save(f"{OUT}/oof_{cfg['tag']}.npy", oof); np.save(f"{OUT}/test_{cfg['tag']}.npy", tp)
    json.dump({"macro": float(macro)}, open(f"{OUT}/cv_{cfg['tag']}.json", "w"))
    log("saved", cfg["tag"])


if __name__ == "__main__":
    main()
