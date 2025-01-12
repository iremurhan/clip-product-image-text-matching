import torch
from tqdm import tqdm


def train_model(classifier, train_loader, criterion, optimizer, device, config):
    """
    Training loop for the CLIP fine-tuning.
    """
    classifier.train()
    total_loss, correct, total = 0, 0, 0

    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        logits = classifier(inputs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % config["model"]["save_every_n_steps"] == 0:
            torch.save(classifier.state_dict(), "classifier_checkpoint.pth")
            print(f"Checkpoint at step {step+1}")

    accuracy = correct / total
    print(f"Training completed - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")
