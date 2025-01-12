import yaml
import json
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from src.data_preparation import prepare_data
from src.dataset import CLIPDataset
from src.models import CLIPClassifier
from src.training import train_model

# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Prepare data
train_path, test_path = prepare_data(config)

# Load data
processor = CLIPProcessor.from_pretrained(config["model"]["pretrained_model"])
with open(train_path, "r") as f:
    train_pairs = json.load(f)
train_dataset = CLIPDataset(train_pairs, processor)
train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

# Initialize model
clip_model = CLIPModel.from_pretrained(config["model"]["pretrained_model"])
classifier = CLIPClassifier(clip_model).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Train model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=config["model"]["learning_rate"])
train_model(classifier, train_loader, criterion, optimizer, torch.device("cuda" if torch.cuda.is_available() else "cpu"), config)
