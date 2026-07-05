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

1. **EDA & redaction analysis** (`scripts/`) — understand the two redaction styles
   (length-preserving star masks and skeleton+digit tokens) and confirm the target
   is genuine fuzzy cross-lingual alignment (candidates re-redacted vs. context).
2. **CV harness & feature baseline** (`src/`) — grouped stratified CV with the exact
   balanced-macro metric; char-n-gram similarity + context features + LightGBM ranker.
3. **Transformer fine-tuning on Kaggle P100** — multilingual multiple-choice models
   (mDeBERTa-v3, XLM-R) with k-fold, saving OOF + test probabilities.
4. **Ensemble & submission** — blend model probabilities weighted by CV, calibrate,
   emit `submission.csv`.

## Layout

- `scripts/` — one-off EDA / probes
- `src/` — reusable library (metric, CV, features, data prep)
- `kaggle/` — kernel scripts run on Kaggle GPU
- `submissions/` — generated submissions (tracked)
- `oof/`, `artifacts/` — CV predictions & models (not tracked)
