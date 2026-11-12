"""Command-line entry point.

    python main.py prepare
    python main.py train --loss ce|contrastive
    python main.py evaluate --model zero-shot|ce|contrastive [--checkpoint PATH]
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from src.data_preparation import prepare_data
from src.dataset import CLIPPairDataset, load_pairs
from src.evaluation import predict, summarize
from src.models import CLIPPairClassifier, CLIPSimilarityModel
from src.training import train
from src.utils import get_device, load_config, set_seed


def build_model(kind, config, device, checkpoint=None):
    clip = CLIPModel.from_pretrained(config["model"]["pretrained_model"])
    model = CLIPPairClassifier(clip) if kind == "ce" else CLIPSimilarityModel(clip)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device)


def make_loader(path, processor, config, shuffle):
    dataset = CLIPPairDataset(load_pairs(path), processor)
    return DataLoader(dataset, batch_size=config["model"]["batch_size"], shuffle=shuffle,
                      num_workers=config.get("num_workers", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="build pairs and the image-disjoint train/test split")
    p_train = sub.add_parser("train", help="fine-tune CLIP")
    p_train.add_argument("--loss", choices=["ce", "contrastive"], required=True)
    p_eval = sub.add_parser("evaluate", help="evaluate on the test split")
    p_eval.add_argument("--model", choices=["zero-shot", "ce", "contrastive"], required=True)
    p_eval.add_argument("--checkpoint")
    p_eval.add_argument("--threshold", type=float,
                        help="cosine-similarity threshold (overrides the config); "
                             "zero-shot CLIP needs a much lower value than a fine-tuned model")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    if args.command == "prepare":
        prepare_data(config)
        return

    device = get_device()
    processor = CLIPProcessor.from_pretrained(config["model"]["pretrained_model"])

    if args.command == "train":
        model = build_model(args.loss, config, device)
        train(model, make_loader(config["data"]["train_pairs"], processor, config, True),
              args.loss, config, device)
        return

    if args.model != "zero-shot" and not args.checkpoint:
        parser.error("--checkpoint is required for fine-tuned models")
    mode = "ce" if args.model == "ce" else "similarity"
    model = build_model(mode, config, device, args.checkpoint)
    threshold = args.threshold if args.threshold is not None else config["evaluation"]["threshold"]
    test_loader = make_loader(config["data"]["test_pairs"], processor, config, False)
    labels, preds, scores = predict(model, test_loader, mode, device, threshold=threshold)
    result = summarize(labels, preds, scores)
    print(f"accuracy {result['accuracy']:.4f}  macro-F1 {result['macro_f1']:.4f}  "
          f"ROC-AUC {result['roc_auc']:.4f}")
    print(result["report"])


if __name__ == "__main__":
    main()
