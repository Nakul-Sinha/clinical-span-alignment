# Clinical Span Alignment

## The problem

I get a redacted Spanish clinical note with one concept marked inside anchor
tags, and a redacted note in a target language. I have to pick which of four
candidate spans in the target note refers to the same concept. Six target
languages, Czech, English, Italian, Dutch, Romanian and Swedish, and three entity
types, disease, symptom and procedure. Scoring is balanced macro accuracy over
all 18 language and entity groups, so a language I handle badly cannot be hidden
behind one I handle well.

The catch is the redaction. Both sides are masked, in two different styles,
length preserving star masks and skeleton plus digit tokens, so I am aligning
concepts I cannot fully read.

## What I did

Fine-tuned multilingual encoders only, no TF-IDF or n-gram shortcuts, since this
is meant to be a fine-tuning task rather than generic text classification.

I fine-tune XLM-R and mDeBERTa-v3 as multiple choice models, with a custom pair
scoring head over the source plus anchor against each candidate, and special
tokens marking the anchor. Then I blend the folds weighted by out-of-fold
balanced macro accuracy.

The real difficulty turned out to be training stability rather than architecture.
Naive fine-tuning collapses on a fraction of folds, with the loss stuck at ln 4
and the model emitting uniform predictions. Three things fixed it: warming up the
head with the backbone frozen for the first steps, selecting the best checkpoint
rather than the last, and automatically retrying a fold when it collapses.

Baseline is 0.67, I am around 0.73, and I am aiming past 0.80.

## Layout

`src/clinspan` holds the CV harness, `scripts/` the redaction analysis, and
`kaggle/train_mc.py` the training run. `TECHNICAL.md` has the detail. Datasets
are not committed.
