# Methodology & Findings — LSTM Hoax Classification, Revisited

This document explains what changed between the original published pipeline
(`klasifikasi-hoax-lstm-glove`, archived) and this revised pipeline, why each change was made,
and what it demonstrates.

## 1. Background

The original paper — Sunan, Erliawan K., & Aditya (2024), *"Klasifikasi Hoax Berita Politik
Menggunakan Algoritma LSTM dengan Penambahan Fitur Embedding GloVe"*, JEPIN 10(2), 287–295,
[doi:10.26418/jp.v10i2.76042](https://doi.org/10.26418/jp.v10i2.76042) — reported 99.9% training
accuracy and 98.4% validation accuracy for an LSTM model augmented with GloVe embeddings, on a
dataset of 9,534 Indonesian political news articles labeled `HOAX` or `VALID`.

A post-publication code review (August 2026) found several methodological issues that call the
paper's central claim into question. This repository is a corrected re-implementation, built to
(a) fix those issues and (b) honestly report how much the results change once they're fixed.

## 2. Issues found in the original pipeline

### 2.1 GloVe embedding is English, not Indonesian

`glove.6B.100d.txt` is Stanford NLP's English-language release (Wikipedia 2014 + Gigaword 5).
The dataset's text is Indonesian (post-stemming). A vocabulary coverage check found that the
large majority of the dataset's ~5,000-word vocabulary is **not present** in the English GloVe
dictionary, so those words default to a **zero vector**. Because the embedding layer was also
configured with `trainable=False`, those zero vectors are never updated during training — the
words are effectively invisible to the model for the entire training process.

**Fix:** use FastText's official Indonesian model (`cc.id.300.bin`), which — unlike GloVe —
represents out-of-vocabulary words via character n-gram (subword) composition, so even
stemmed/non-standard word forms get a meaningful, non-zero vector. The embedding layer is also
set to `trainable=True` so the model can further fine-tune these vectors on the task.

### 2.2 SMOTE applied before train/test split (data leakage)

```python
# original notebook, order of operations:
smote = SMOTE(random_state=42, k_neighbors=4)
padded_sequences_res, integer_categories_res = smote.fit_resample(padded_sequences, integer_categories)
# ... train_test_split() happens AFTER this
```

SMOTE generates synthetic samples by interpolating between nearest neighbors across the
**entire** dataset. Because this happened before the train/test split, correlated synthetic
samples could end up distributed across train, validation, and test sets — meaning the reported
"test" performance is not a clean measurement of generalization to unseen data.

**Fix:** split the data first; apply any class-imbalance handling only to the training set.

### 2.3 SMOTE applied to categorical token IDs

SMOTE performs linear interpolation between neighboring data points — appropriate for continuous
numeric features, not for token IDs, which are categorical/nominal identifiers. Interpolating
between token ID 245 and token ID 812 produces a "token" that does not correspond to any real
word. This means the synthetic minority-class (`HOAX`) samples used in training likely consisted
partly of token sequences that are not real language — which risks teaching the model to detect
"this looks like scrambled tokens" as a shortcut for the `HOAX` label, rather than learning
genuine linguistic patterns of hoax content.

**Fix:** replace SMOTE with a class-weighted loss function (`nn.CrossEntropyLoss(weight=...)`),
computed from the training set's class distribution. This handles imbalance without generating
synthetic sequences.

### 2.4 Padding length correlates with label

The original notebook padded every sequence to `max(len(seq) for seq in sequences)` — the
single longest sequence in the **entire dataset**. A length audit
(see `notebooks/01_data_audit.ipynb`) found that `HOAX` articles are on average much longer and
far more variable in length (mean ≈ 334 words, max 3,530) than `VALID` articles (mean ≈ 188
words, max 1,351). Because Keras `pad_sequences` pre-pads by default, this means `VALID`
sequences ended up with substantially more leading zero-padding than `HOAX` sequences — a
trivial, non-semantic signal an LSTM could exploit as a shortcut.

**Fix:** cap `max_len` at the 95th percentile of **training-set** sequence lengths, with
truncation for longer sequences, rather than the dataset-wide maximum.

### 2.5 (Documented, not "fixed") Source-based dataset bias — including a strong boilerplate leak

Every `VALID` article comes from detik.com and every `HOAX` article comes from turnbackhoax.id.
No leftover URLs, HTML entities, or site-name strings were found in the cleaned text (see
`notebooks/01_data_audit.ipynb`, Section 3) — but a first-word audit found something more
concrete and more serious than a general style difference:

**41.7% of `HOAX` articles (542/1,300) begin with the word "hasil"** — the stemmed remnant of
turnbackhoax.id/MAFINDO's fact-check article template ("Hasil periksa fakta [checker name]
[institution]..."). Only 0.2% (20/8,234) of `VALID` articles start with that word. A trivial
rule — "if the text starts with 'hasil', predict HOAX" — would already be ~96.5% precise on the
subset of articles where it fires, without understanding any actual content.

**Implication:** part of the high accuracy reported by both the original model and any revised
model trained on this dataset (including the FastText/PyTorch model in this repo) may reflect
detecting this site template rather than semantic understanding of what makes content
misleading. This is a property of the dataset itself, not something fixed by changing the
embedding, framework, or imbalance-handling strategy — it would require re-scraping with the
template stripped, or sourcing `HOAX` examples from more than one site. It's disclosed here so
results aren't over-interpreted as "the model understands what makes news fake" versus "the
model partly distinguishes two publishers' article templates."

## 3. What did NOT need fixing

The dataset's `desc` column was found to already be fully cleaned (no URLs, HTML entities,
uppercase characters, or punctuation) and stemmed (consistent with the Sastrawi + Enhanced
Confix Stripping pipeline described in the paper). No re-cleaning or re-stemming was performed —
see `notebooks/01_data_audit.ipynb` Section 3 for the verification.

## 4. Architecture changes

| | Original | Revised |
|---|---|---|
| Framework | TensorFlow / Keras | PyTorch |
| Embedding | GloVe 100d, English, frozen | FastText 300d, Indonesian, fine-tuned |
| LSTM | 2-layer, unidirectional (64 → 32 units) | 2-layer, **bidirectional** (64 → 32 units) |
| Sequence length | Dataset-wide max | 95th percentile of training set + truncation |
| Imbalance handling | SMOTE (before split, on token IDs) | Class-weighted loss (training set only) |

The bidirectional LSTM change is an additional optimization enabled by the larger, more
meaningful 300-dimensional embeddings — it lets the model use context from both directions of
each sentence, which is generally beneficial for text classification.

## 5. Results

See `notebooks/03_before_after_comparison.ipynb` for the full comparison, and
`docs/results_revised.json` for the raw numbers produced by `notebooks/02_LSTM_FastText_PyTorch.ipynb`.

**Important:** if the revised model's accuracy comes out *lower* than the original paper's
claimed 98.04% test accuracy, that is not necessarily a regression — see Section 2 above.
Removing data leakage and shortcut signals is expected to reduce inflated performance figures
and produce a number that better reflects real-world generalization.

## 6. Citation

If you build on this repository, please note it is an independent, unaffiliated revision of the
original published work:

> Sunan, R. A., K., H. F. E., & Aditya, C. S. K. (2024). Klasifikasi Hoax Berita Politik
> Menggunakan Algoritma Long Short-Term Memory (LSTM) dengan Penambahan Fitur Embedding Global
> Vector (GloVe). *Jurnal Edukasi dan Penelitian Informatika (JEPIN)*, 10(2), 287–295.
> https://doi.org/10.26418/jp.v10i2.76042

The original paper is licensed CC BY-NC-SA 4.0 by its publisher; this repository's code is
licensed separately (MIT — see `LICENSE`).
