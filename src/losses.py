import torch

from src.data_preparation import ALIGNED


def contrastive_pair_loss(similarity, labels, margin=0.2):
    """Margin-based contrastive loss on image-caption cosine similarity.

    Aligned pairs (label 0) are pulled towards similarity 1; misaligned pairs
    (label 1) are penalised only when their similarity exceeds `margin`.

    The original notebook passed the dataset labels (0 = aligned) into a loss
    that expected 1 = aligned, so it pulled *misaligned* pairs together. The
    explicit conversion below fixes that.
    """
    is_aligned = (labels == ALIGNED).float()
    positive = is_aligned * (1 - similarity).pow(2)
    negative = (1 - is_aligned) * torch.clamp(similarity - margin, min=0).pow(2)
    return (positive + negative).mean()
