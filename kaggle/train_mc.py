"""Cross-lingual clinical span anchoring — multilingual multiple-choice fine-tuning.

Self-contained Kaggle kernel script. Reads the challenge CSVs from the attached
dataset, trains an AutoModelForMultipleChoice with deterministic stratified k-fold
CV, and writes OOF + test probabilities, per-group CV, and a submission.csv.

Config is injected by the local pusher via the CONFIG block below.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# === CONFIG_JSON_START ===
CONFIG = {
    "model_name": "xlm-roberta-base",
    "tag": "xlmr-base",
    "max_len": 256,
    "tgt_chars": 320,
    "epochs": 3,
    "lr": 2e-5,
    "batch_size": 8,
    "grad_accum": 2,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "n_splits": 5,
    "seed": 42,
    "folds_to_run": [0, 1, 2, 3, 4],
    "max_grad_norm": 1.0,
    "tta": True,
    "subset": 0,
}
# === CONFIG_JSON_END ===
# ----------------------------------------------------------------------------

INPUT = "/kaggle/input/clinspan-data"
OUT = "/kaggle/working"
CANDS = ["candidate_a", "candidate_b", "candidate_c", "candidate_d"]
LETTERS = ["A", "B", "C", "D"]
ANCHOR_TOKENS = ["[[ANCHOR]]", "[[/ANCHOR]]"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def ensure_deps():
    try:
        import transformers  # noqa
        v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        assert v >= (4, 38)
    except Exception:
        os.system("pip install -q -U 'transformers>=4.40' sentencepiece accelerate")


def det_folds(df, n_splits, seed):
    """Deterministic stratified folds by (language, entity), reproducible anywhere."""
    def h(s):
        return int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16)
    df = df.copy()
    df["_h"] = df["id"].map(h)
    df["fold"] = -1
    for _, idx in df.groupby(["target_language", "entity_type"]).groups.items():
        sub = df.loc[idx].sort_values("_h")
        df.loc[sub.index, "fold"] = np.arange(len(sub)) % n_splits
    return df["fold"].to_numpy()


def make_context(row, tgt_chars):
    hint = f"language: {row.target_language} | entity: {row.entity_type}"
    ctx = f"{hint} | source: {row.source_context}"
    if tgt_chars > 0:
        ctx = f"{ctx} | target: {str(row.target_context)[:tgt_chars]}"
    return ctx


def main():
    ensure_deps()
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup

    cfg = CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device:", device, "| torch:", torch.__version__)
    log("config:", json.dumps(cfg))

    def set_seed(s):
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    set_seed(cfg["seed"])

    import glob
    log("input tree:", glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")[:20])
    def find_csv(name):
        hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if not hits:
            raise FileNotFoundError(f"{name} not found under /kaggle/input")
        return sorted(hits, key=len)[0]
    train = pd.read_csv(find_csv("train.csv"))
    test = pd.read_csv(find_csv("test.csv"))
    if cfg["subset"]:
        train = train.groupby(["target_language", "entity_type"], group_keys=False).head(
            max(4, cfg["subset"] // 18)).reset_index(drop=True)
        test = test.head(cfg["subset"]).reset_index(drop=True)
    log("shapes:", train.shape, test.shape)

    folds = det_folds(train, cfg["n_splits"], cfg["seed"])
    y = train["selected_option"].map({l: i for i, l in enumerate(LETTERS)}).to_numpy()

    train["_ctx"] = [make_context(r, cfg["tgt_chars"]) for r in train.itertuples()]
    test["_ctx"] = [make_context(r, cfg["tgt_chars"]) for r in test.itertuples()]

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.add_special_tokens({"additional_special_tokens": ANCHOR_TOKENS})

    class MCData(Dataset):
        def __init__(self, df, order=(0, 1, 2, 3), labels=None):
            self.ctx = df["_ctx"].tolist()
            self.cands = [[str(df.iloc[i][CANDS[o]]) for o in order] for i in range(len(df))]
            self.labels = labels if labels is not None else [-1] * len(df)

        def __len__(self):
            return len(self.ctx)

        def __getitem__(self, i):
            return self.ctx[i], self.cands[i], int(self.labels[i])

    def collate(batch):
        ctxs, cand_lists, labs = zip(*batch)
        first, second = [], []
        for ctx, cands in zip(ctxs, cand_lists):
            for c in cands:
                first.append(ctx)
                second.append(c)
        enc = tok(first, second, truncation="longest_first", max_length=cfg["max_len"],
                  padding=True, return_tensors="pt")
        b = len(ctxs)
        out = {k: v.view(b, 4, -1) for k, v in enc.items()}
        out["labels"] = torch.tensor(labs, dtype=torch.long)
        return out

    class MCModel(nn.Module):
        """Custom multiple-choice head on a base encoder (version-robust across model types).

        Encodes each (context, candidate) pair, mask-aware mean-pools the token states,
        and scores each of the 4 choices with a shared linear layer -> (B, 4) logits.
        """
        def __init__(self, model_name, vocab_size, pool="mean", p_drop=0.1):
            super().__init__()
            # force fp32 master weights (some checkpoints, e.g. mDeBERTa, ship as fp16
            # which breaks GradScaler.unscale_); autocast still does fp16 compute.
            self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
            self.backbone.resize_token_embeddings(vocab_size)
            if cfg.get("grad_ckpt"):
                self.backbone.gradient_checkpointing_enable()
                if hasattr(self.backbone, "config"):
                    self.backbone.config.use_cache = False
            h = self.backbone.config.hidden_size
            self.pool = pool
            self.dropout = nn.Dropout(p_drop)
            self.classifier = nn.Linear(h, 1)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            B, C, L = input_ids.shape
            kwargs = dict(input_ids=input_ids.view(B * C, L),
                          attention_mask=attention_mask.view(B * C, L))
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids.view(B * C, L)
            out = self.backbone(**kwargs)
            last = out.last_hidden_state                       # (B*C, L, H)
            am = kwargs["attention_mask"].unsqueeze(-1).to(last.dtype)
            if self.pool == "cls":
                pooled = last[:, 0]
            else:
                pooled = (last * am).sum(1) / am.sum(1).clamp(min=1e-6)
            logits = self.classifier(self.dropout(pooled)).view(B, C)
            return logits

    def build_model():
        m = MCModel(cfg["model_name"], len(tok), pool=cfg.get("pool", "mean"))
        return m.to(device)

    @torch.no_grad()
    def predict(model, df, order=(0, 1, 2, 3)):
        model.eval()
        dl = DataLoader(MCData(df, order), batch_size=cfg["batch_size"] * 2,
                        shuffle=False, collate_fn=collate, num_workers=2)
        out = []
        for batch in dl:
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                logits = model(**batch)
            out.append(logits.float().softmax(-1).cpu().numpy())
        probs = np.concatenate(out, 0)
        # undo permutation: probs columns are in permuted order
        inv = np.argsort(order)
        return probs[:, inv]

    import copy
    oof = np.zeros((len(train), 4))
    test_probs = np.zeros((len(test), 4))
    run_folds = cfg["folds_to_run"]
    COLLAPSE = cfg.get("collapse_thresh", 0.32)
    max_attempts = cfg.get("max_attempts", 3)

    def run_fold(fold, seed):
        """Train one fold; keep the best-val checkpoint; return (best_acc, oof_probs, test_probs)."""
        set_seed(seed)
        tr_df = train[folds != fold].reset_index(drop=True)
        va_df = train[folds == fold].reset_index(drop=True)
        va_y = y[folds == fold]

        model = build_model()
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": cfg["weight_decay"]},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(params, lr=cfg["lr"])
        dl = DataLoader(MCData(tr_df, labels=y[folds != fold]), batch_size=cfg["batch_size"],
                        shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
        steps_per_epoch = len(dl) // cfg["grad_accum"]
        total_steps = steps_per_epoch * cfg["epochs"]
        sched = get_linear_schedule_with_warmup(opt, int(total_steps * cfg["warmup_ratio"]), total_steps)
        scaler = torch.amp.GradScaler("cuda")
        lossf = torch.nn.CrossEntropyLoss()

        best_acc, best_state = -1.0, None
        for epoch in range(cfg["epochs"]):
            model.train()
            opt.zero_grad()
            running = 0.0
            t0 = time.time()
            for bi, batch in enumerate(dl):
                lb = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda"):
                    logits = model(**batch)
                    loss = lossf(logits, lb) / cfg["grad_accum"]
                scaler.scale(loss).backward()
                running += loss.item() * cfg["grad_accum"]
                if (bi + 1) % cfg["grad_accum"] == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad()
            acc = (predict(model, va_df).argmax(1) == va_y).mean()
            log(f"    epoch {epoch}: loss={running/len(dl):.4f} val_acc={acc:.4f} ({time.time()-t0:.0f}s)")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        oof_fold = predict(model, va_df)
        tp = predict(model, test)
        del model
        torch.cuda.empty_cache()
        return float(best_acc), oof_fold, tp

    for fold in run_folds:
        log(f"===== fold {fold} =====")
        best_acc, oof_fold, tp = -1.0, None, None
        for attempt in range(max_attempts):
            seed = cfg["seed"] + fold + attempt * 1000
            log(f"  attempt {attempt} (seed {seed})")
            ba, of_, tp_ = run_fold(fold, seed)
            if ba > best_acc:
                best_acc, oof_fold, tp = ba, of_, tp_
            if ba >= COLLAPSE:
                break
            log(f"  fold {fold} collapsed (val_acc={ba:.3f}); retrying")
        oof[folds == fold] = oof_fold
        test_probs += tp / len(run_folds)
        log(f"  fold {fold} accepted best_val_acc={best_acc:.4f}")

    # ---- scoring ----
    ran_mask = np.isin(folds, run_folds)
    oof_pred = oof.argmax(1)
    train["_pred_idx"] = oof_pred
    train["_correct"] = (oof_pred == y)
    groups = []
    for (lang, ent), g in train[ran_mask].groupby(["target_language", "entity_type"]):
        groups.append({"lang": lang, "ent": ent, "n": len(g), "acc": g["_correct"].mean()})
    gdf = pd.DataFrame(groups)
    macro = gdf["acc"].mean()
    plain = train[ran_mask]["_correct"].mean()
    log(f"OOF balanced macro accuracy: {macro:.4f} | plain: {plain:.4f}")
    print(gdf.sort_values("acc").to_string(index=False), flush=True)

    tag = cfg["tag"]
    np.save(f"{OUT}/oof_{tag}.npy", oof)
    np.save(f"{OUT}/test_{tag}.npy", test_probs)
    sub = pd.DataFrame({"id": test["id"], "selected_option": [LETTERS[i] for i in test_probs.argmax(1)]})
    sub.to_csv(f"{OUT}/submission_{tag}.csv", index=False)
    with open(f"{OUT}/cv_{tag}.json", "w") as f:
        json.dump({"macro": float(macro), "plain": float(plain),
                   "groups": gdf.to_dict("records"), "config": cfg}, f, indent=2)
    log("saved outputs for tag:", tag)


if __name__ == "__main__":
    main()
