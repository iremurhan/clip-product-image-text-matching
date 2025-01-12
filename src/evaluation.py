import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


def evaluate_model(classifier, test_loader, device):
    """
    Evaluate the fine-tuned CLIP model.
    """
    classifier.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Aligned", "Misaligned"])
    return accuracy, f1, report
