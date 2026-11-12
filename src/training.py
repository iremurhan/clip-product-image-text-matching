import os

import torch
from tqdm import tqdm

from src.losses import contrastive_pair_loss


def train(model, loader, loss_type, config, device):
    """Fine-tune `model` for config['model']['epochs'] epochs.

    loss_type: "ce" (CLIPPairClassifier + cross-entropy) or
               "contrastive" (CLIPSimilarityModel + contrastive_pair_loss).
    """
    mcfg = config["model"]
    lr = mcfg["lr_ce"] if loss_type == "ce" else mcfg["lr_contrastive"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    ckpt_dir = config["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, mcfg["epochs"] + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for inputs, labels in tqdm(loader, desc=f"[{loss_type}] epoch {epoch}"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            if loss_type == "ce":
                loss = criterion(model(inputs), labels)
            else:
                loss = contrastive_pair_loss(model(inputs), labels, margin=mcfg["margin"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        path = os.path.join(ckpt_dir, f"{loss_type}_epoch{epoch}.pth")
        torch.save(model.state_dict(), path)
        print(f"epoch {epoch}: mean loss {total_loss / max(n_batches, 1):.4f} -> {path}")
