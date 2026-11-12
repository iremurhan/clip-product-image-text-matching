# CLIP Product Image–Text Matching

Fine-tuning CLIP (ViT-B/32) to decide whether a product image and a product title belong together, using two objectives: a cross-entropy pair classifier and a margin-based contrastive loss.

Term project for CMPE 586 (Transformer-Based Pre-Training Methods in NLP), Boğaziçi University, Fall 2024. The original report is in [`docs/report.pdf`](docs/report.pdf).

## Task

Each example is an (image, title) pair labelled **aligned** (the product's own title) or **misaligned** (the title of a random product that shares no category with it). The motivation is flagging listings or user-uploaded photos that do not match the product they are attached to.

## Data

Not included in this repository. Data comes from the public Kaggle dataset [Trendyol Product Dataset](https://www.kaggle.com/datasets/bilalelebi/trendyol-product-dataset) (product links, titles, categories). `notebooks/exploration.ipynb` (section 2) downloads each product's main image from its product page and writes `data/updated_trendyol_dataset_with_images.csv`. The original run produced 23,036 aligned and 23,036 misaligned pairs.

## Method

| Model | What is trained | Prediction |
|---|---|---|
| Zero-shot | nothing | cosine(image, title) > threshold → aligned |
| Cross-entropy | all CLIP weights + a 1→2 linear head on the cosine similarity | argmax of the head |
| Contrastive | all CLIP weights, loss `(1−s)²` for aligned and `max(0, s−m)²` for misaligned pairs (m = 0.2) | cosine > threshold → aligned |

The train/test split is **image-disjoint** (`GroupShuffleSplit` on the image path), because every image appears in both an aligned and a misaligned pair.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py prepare                                   # pairs + image-disjoint split
python main.py train --loss ce                           # or --loss contrastive
python main.py evaluate --model ce --checkpoint checkpoints/ce_epoch3.pth
python main.py evaluate --model contrastive --checkpoint checkpoints/contrastive_epoch3.pth
python main.py evaluate --model zero-shot --threshold 0.25
```

Evaluation reports accuracy, macro-F1 and ROC-AUC (threshold-independent, so the zero-shot model can be compared fairly). Settings live in `configs/config.yaml`. Checkpoints are written to `checkpoints/`, which is git-ignored.

```bash
pytest -q    # unit tests; no images, network or CLIP weights needed
```

## Results

**Original course run (Jan 2025, notebook):** these numbers are kept for the record, but they are **not a valid measure of generalisation**:

| Model | Accuracy | F1 |
|---|---|---|
| Cross-entropy | 0.848 | 0.836 (binary, misaligned class) |
| Contrastive | 0.790 | 0.790 (weighted) |

Known problems with that run, found when this repository was cleaned up:

1. **Train/test leakage.** The split was random over pairs, so 90.5% of test images (and 78.8% of test titles) also appeared in training.
2. **Inverted contrastive labels.** The dataset used 0 = aligned, while the loss expected 1 = aligned. The contrastive model therefore learned to make *misaligned* pairs similar, and its threshold rule was inverted to match. Its accuracy is real, but it does not reflect better image–text alignment.
3. **Metrics are not comparable.** The two F1 scores use different averaging, and the zero-shot baseline listed in the report is not in the notebook.

All three are fixed in the current code. Results on the image-disjoint split have **not been re-run yet**.

## Repository layout

```
main.py                 CLI: prepare / train / evaluate
configs/config.yaml     paths and hyper-parameters
src/                    data preparation, dataset, models, loss, training, evaluation
tests/                  unit tests
notebooks/              original exploration notebook (data collection, first experiments)
docs/report.pdf         course report
```
