# **FineTuneCLIP-Ecommerce: Detecting Misaligned Product Images in E-Commerce Reviews**

## **Overview**
`FineTuneCLIP-Ecommerce` is a project focused on fine-tuning the pre-trained CLIP (Contrastive Language–Image Pretraining) model to detect misaligned images in the context of e-commerce platforms. By using a curated dataset of product images and user-uploaded review images, this project demonstrates how multi-modal models like CLIP can be adapted for real-world data.

The experiments were initially conducted in Jupyter notebooks for quick iteration and analysis. Afterward, the project structure was reorganized into a modular format for improved maintainability and scalability.

---

## **Features**
- **Fine-tuning CLIP**: Models fine-tuned with both contrastive and cross-entropy losses to detect semantic misalignment.
- **Noisy Dataset Handling**: Dataset curated from Trendyol product links, combining official product images with user-uploaded review images.
- **Comprehensive Evaluation**: Metrics such as accuracy, F1-score, and classification reports to assess performance.
- **Modular Design**: Project refactored into multiple Python modules for better organization and usability.

---

## **Directory Structure**
```plaintext
FineTuneCLIP-Ecommerce/
│
├── .venv/                              # Virtual environment folder (ignored in version control)
│
├── checkpoints/                        # Saved model checkpoints
│   ├── clip_checkpoint.pth
│   ├── clip_fine_tune_checkpoint.pth
│   └── clip_fine_tuned_epoch_3.pth
│
├── configs/                            # Configuration files
│   └── config.yaml
│
├── data/                               # Dataset and processed JSON files
│   ├── aligned_pairs.json
│   ├── misaligned_pairs.json
│   ├── train_pairs.json
│   ├── test_pairs.json
│   └── updated_trendyol_dataset_with_images.csv
│
├── notebooks/                          # Jupyter notebooks for experimentation
│   └── exploration.ipynb
│
├── src/                                # Python modules for the project
│   ├── __init__.py                     # Package initialization
│   ├── data_preparation.py             # Dataset preparation logic
│   ├── dataset.py                      # Dataset class and data loaders
│   ├── evaluation.py                   # Evaluation and metrics
│   ├── models.py                       # Model architecture definitions
│   ├── training.py                     # Training logic
│   └── utils.py                        # Helper functions (e.g., config loading)
│
├── main.py                             # Main script for running the project
├── README.md                           # Documentation (this file)
└── requirements.txt                    # Python dependencies
```

---

## **Installation and Setup**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/iremurhan/FineTuneCLIP-Ecommerce.git
   cd FineTuneCLIP-Ecommerce
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Pretrained CLIP Model**
   ```bash
   transformers-cli download openai/clip-vit-base-patch32
   ```

---

## **Configuration**

All configurable paths and parameters are stored in `configs/config.yaml`. Update this file to customize your project settings.

**Example Configuration:**
```yaml
data:
  product_image_dir: "data/product_images"
  review_image_dir: "data/review_images"
  dataset_file: "data/updated_trendyol_dataset_with_images.csv"
  aligned_pairs: "data/aligned_pairs.json"
  misaligned_pairs: "data/misaligned_pairs.json"
  train_pairs: "data/train_pairs.json"
  test_pairs: "data/test_pairs.json"

model:
  pretrained_model: "openai/clip-vit-base-patch32"
  batch_size: 32
  learning_rate: 1e-5
  epochs: 3

evaluation:
  threshold: 0.4
```

---

## **Usage**

1. **Prepare the Dataset**
   Prepare aligned and misaligned pairs, then split them into train/test sets:
   ```bash
   python main.py --prepare-data
   ```

2. **Train the Model**
   Fine-tune the CLIP model:
   ```bash
   python main.py --train
   ```

3. **Evaluate the Model**
   Run evaluation on the test dataset:
   ```bash
   python main.py --evaluate
   ```

---

## **Experiments**

All initial experiments were conducted in the Jupyter notebook located in the `notebooks/` folder:

- **`exploration.ipynb`**: Includes early experimentation with fine-tuning and evaluation using contrastive and cross-entropy loss functions.

After completing the notebook-based experiments, the project was modularized into Python scripts for better scalability and maintainability.

---

## **Results**

The table below summarizes the results from the fine-tuning experiments:

| Model                          | Accuracy | F1-Score | Precision (Aligned/Misaligned) | Recall (Aligned/Misaligned) |
|--------------------------------|----------|----------|---------------------------------|-----------------------------|
| Original CLIP                 | 0.68     | 0.65     | 0.70 / 0.65                    | 0.60 / 0.70                |
| Fine-Tuned CLIP (Contrastive) | 0.79     | 0.79     | 0.76 / 0.82                    | 0.84 / 0.74                |
| Fine-Tuned CLIP (Cross-Entropy) | 0.85    | 0.84     | 0.80 / 0.91                    | 0.92 / 0.78                |

### **Observations:**
- Fine-tuning with cross-entropy loss provided the best overall performance for binary classification.
- Contrastive loss performed slightly worse but retained robust embedding alignment for similarity-based tasks.

---

## **Acknowledgments**

This project uses the Trendyol Product Dataset for experiments.
This is the term project of the CMPE 586 Sp. Tp. Transformer-Based Pretraining Methods in Natural Language Processing, lectured by Onur Güngör at Bogazici University Computer Engineering Department.

---

## **Contributors**

- **Beyza İrem Urhan**  
  Bogazici University  
  [beyza.urhan@bogazici.edu.tr](mailto:beyza.urhan@bogazici.edu.tr)