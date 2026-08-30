# Political Hoax News Classification — Revisited (LSTM + FastText)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)](https://pytorch.org)

> **Status: Active revision.** This is an independent, corrected re-implementation of
> [`klasifikasi-hoax-lstm-glove`](https://github.com/rohid001/klasifikasi-hoax-lstm-glove.git) (archived original), built to fix methodological issues
> found in that repository's published pipeline. See
> [`docs/METHODOLOGY_AND_FINDINGS.md`](docs/METHODOLOGY_AND_FINDINGS.md) for the full writeup.

## Why this repository exists

The original project reproduced the code behind a published paper (JEPIN Vol.10 No.2, 2024,
[doi:10.26418/jp.v10i2.76042](https://doi.org/10.26418/jp.v10i2.76042)) claiming 98%+ accuracy
for LSTM + GloVe hoax classification. A post-publication review found:

1. The GloVe embedding used was **English**, not Indonesian — most of the dataset's vocabulary
   got zero vectors, frozen (`trainable=False`) for the entire training process.
2. SMOTE was applied **before** the train/test split, on raw token IDs — a data leakage risk
   plus synthetic sequences that don't correspond to real language.
3. Sequence padding length correlated with class label — a potential shortcut signal unrelated
   to actual content.

This repository fixes all three issues and documents, with before/after numbers, how much they
mattered. Full technical rationale: [`docs/METHODOLOGY_AND_FINDINGS.md`](docs/METHODOLOGY_AND_FINDINGS.md).

## What changed

| | Original | Revised (this repo) |
|---|---|---|
| Framework | TensorFlow/Keras | **PyTorch** |
| Embedding | GloVe 100d, English, frozen | **FastText 300d, Indonesian**, subword-aware, fine-tuned |
| LSTM | 2-layer unidirectional | 2-layer **bidirectional** |
| Imbalance handling | SMOTE (before split, on token IDs) | **Class-weighted loss** (training set only) |
| Sequence length | Dataset-wide maximum | 95th percentile of training set + truncation |

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── 01_data_audit.ipynb                  # Dataset audit: is more preprocessing needed?
│   │                                         # Also documents the SMOTE/padding issues found.
│   ├── 02_LSTM_FastText_PyTorch.ipynb        # Main model: setup, training, evaluation
│   └── 03_before_after_comparison.ipynb      # Original vs revised results, error analysis
├── data/
│   └── dataset_merge_v2.csv                  # Same dataset as the original repo
├── models/                                   # Created by notebook 02 (checkpoint, vocab, metadata)
└── docs/
    └── METHODOLOGY_AND_FINDINGS.md           # Full technical writeup
```

## Running this project

1. Open `notebooks/01_data_audit.ipynb` first (Google Colab recommended) to see the dataset
   audit and understand the design decisions made in notebook 02.
2. Run `notebooks/02_LSTM_FastText_PyTorch.ipynb`. This notebook:
   - Installs all dependencies (**including PyTorch**) in its first cells — no manual setup
     needed.
   - Downloads the official FastText Indonesian model (`cc.id.300.bin`, several GB — GPU
     runtime recommended).
   - Trains and evaluates the model, saving results to `docs/results_revised.json` and the
     checkpoint to `models/lstm_fasttext_best.pt`.
3. Run `notebooks/03_before_after_comparison.ipynb` to generate the before/after comparison
   (original GloVe model vs this revised model), including a qualitative FastText vs GloVe
   embedding comparison and error analysis.

Or install locally:
```bash
pip install -r requirements.txt
```

## Dataset

Same dataset as the [original repository](#) — 9,534 Indonesian political news articles
(8,234 `VALID` from detik.com, 1,300 `HOAX` from turnbackhoax.id), already cleaned and stemmed.
See `notebooks/01_data_audit.ipynb` for the full data quality audit, including a significant
documented limitation: **41.7% of `HOAX` articles begin with the same fact-check site
boilerplate** ("hasil periksa fakta...", vs 0.2% of `VALID` articles). Results should not be
over-interpreted as "detecting hoax content" in isolation from "detecting one publisher's
article template" — see `docs/METHODOLOGY_AND_FINDINGS.md` Section 2.5.

## Results

See [`docs/METHODOLOGY_AND_FINDINGS.md`](docs/METHODOLOGY_AND_FINDINGS.md) Section 5 and
`notebooks/03_before_after_comparison.ipynb` for the full before/after comparison once the
notebooks have been run.

## Relationship to the original paper

This is an independent, unaffiliated technical revision — not a new version of the published
paper itself. If you use this code, please cite the original paper:

> Sunan, R. A., K., H. F. E., & Aditya, C. S. K. (2024). Klasifikasi Hoax Berita Politik
> Menggunakan Algoritma Long Short-Term Memory (LSTM) dengan Penambahan Fitur Embedding Global
> Vector (GloVe). *Jurnal Edukasi dan Penelitian Informatika (JEPIN)*, 10(2), 287–295.
> https://doi.org/10.26418/jp.v10i2.76042

## Author

- Rohid Aji Sunan — [GitHub: rohid001](https://github.com/rohid001)

## License

Code in this repository: [MIT License](LICENSE). The original paper (referenced, not
reproduced here) remains under JEPIN's CC BY-NC-SA 4.0 license.
