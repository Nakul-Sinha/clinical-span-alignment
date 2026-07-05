# Cross-Lingual Clinical Span Anchoring

Solution for the Cross-Lingual Clinical Span Anchoring challenge: given a redacted
Spanish clinical context with one marked concept (`[[ANCHOR]]…[[/ANCHOR]]`) and a
redacted target-language context, select which of four redacted candidate spans
(A/B/C/D) aligns to the marked Spanish concept.

- **Languages:** Czech (`cz`), English (`en`), Italian (`it`), Dutch (`nl`), Romanian (`ro`), Swedish (`sv`)
- **Entity types:** `disease`, `symptom`, `procedure`
- **Metric:** balanced macro accuracy — accuracy per `(target_language, entity_type)` group, averaged over all 18 groups.
- **Baselines:** AI baseline 0.67; current best ~0.73; target > 0.80.

## Data (not committed)

Local dataset lives outside the repo. Files: `train.csv` (8,406 rows), `test.csv`
(3,464 rows), `sample_submission.csv`, plus `.jsonl` mirrors.

> Note: the challenge description quotes 8,289 / 5,574 rows, but the delivered files
> contain 8,406 / 3,464. The delivered files are authoritative; the submission has
> 3,464 rows.

## Approach

This is a **multilingual fine-tuning** task, so the solution is an ensemble of
fine-tuned multilingual encoders only — no TF-IDF / n-gram / statistical
text-classification shortcuts.

1. **EDA & redaction analysis** (`scripts/`) — the two redaction styles
   (length-preserving star masks and skeleton+digit tokens); confirm genuine fuzzy
   cross-lingual alignment (candidates re-redacted vs. context); no note-level leakage.
2. **CV harness** (`src/clinspan`) — grouped stratified CV with the exact balanced-macro
   metric and per-group breakdown.
3. **Multiple-choice fine-tuning on Kaggle T4** (`kaggle/train_mc.py`) — XLM-R /
   mDeBERTa-v3 with a custom pair-scoring head over `(source+anchor, candidate)`;
   anchor special tokens; fp16; k-fold OOF + test probabilities.
   - **Training stability** was the crux: naive fine-tuning collapses on a fraction of
     folds (loss stuck at ln 4, uniform predictions). Fixes: **head-warmup** (freeze the
     backbone for the first steps so the head settles), best-checkpoint selection, and
     auto-retry-on-collapse.
4. **Ensemble & submission** — OOF-balanced-macro-weighted blend of the fine-tuned
   models; emit `submission.csv`.

> Note: an earlier char-n-gram/TF-IDF feature model was explored for analysis but is
> **excluded from the solution** — the challenge is a fine-tuning task, not generic
> text classification.

## Layout

- `scripts/` — one-off EDA / probes
- `src/` — reusable library (metric, CV, features, data prep)
- `kaggle/` — kernel scripts run on Kaggle GPU
- `submissions/` — generated submissions (tracked)
- `oof/`, `artifacts/` — CV predictions & models (not tracked)
