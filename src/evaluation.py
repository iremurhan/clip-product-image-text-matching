import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from tqdm import tqdm

from src.models import similarity_to_label


@torch.no_grad()
def predict(model, loader, mode, device, threshold=None):
    """Return (labels, predictions, misalignment scores).

    mode "ce": argmax over classifier logits; score = P(misaligned).
    mode "similarity": threshold the image-caption cosine similarity;
                       score = -similarity.
    """
    model.eval()
    labels_all, preds_all, scores_all = [], [], []
    for inputs, labels in tqdm(loader, desc=f"evaluate [{mode}]"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(inputs)
        if mode == "ce":
            preds, scores = out.argmax(dim=1), out.softmax(dim=1)[:, 1]
        else:
            preds, scores = similarity_to_label(out, threshold), -out
        labels_all.extend(labels.tolist())
        preds_all.extend(preds.cpu().tolist())
        scores_all.extend(scores.cpu().tolist())
    return labels_all, preds_all, scores_all


def summarize(labels, preds, scores):
    """Same metrics for every model; ROC-AUC does not depend on the threshold."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "roc_auc": roc_auc_score(labels, scores),
        "report": classification_report(labels, preds, target_names=["aligned", "misaligned"],
                                        digits=3, zero_division=0),
    }
