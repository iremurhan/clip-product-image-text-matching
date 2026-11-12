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

## Evaluation protocol

- **Image-disjoint split.** Each product image occurs in one aligned and one misaligned pair, so pairs are grouped by image before splitting. No test image is seen during training.
- **Label convention.** 0 = aligned, 1 = misaligned throughout data preparation, both losses and evaluation; the contrastive loss pulls aligned pairs together and pushes misaligned pairs below the margin.
- **Comparable metrics.** All three models are scored with accuracy, macro-F1 and ROC-AUC on the same test split. ROC-AUC uses the continuous score of each model, so the zero-shot baseline does not depend on a hand-picked threshold.

The original course notebook (`notebooks/`, Jan 2025) split pairs at random, so most test images also appeared in training, and its contrastive model used the opposite label convention. Its scores, reported in `docs/report.pdf`, are therefore not comparable with this code.

## Repository layout

```
main.py                 CLI: prepare / train / evaluate
configs/config.yaml     paths and hyper-parameters
src/                    data preparation, dataset, models, loss, training, evaluation
tests/                  unit tests
notebooks/              original exploration notebook (data collection, first experiments)
docs/report.pdf         course report
```
