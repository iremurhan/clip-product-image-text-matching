import torch
import torch.nn as nn
import torch.nn.functional as F


def pair_similarity(clip_model, inputs):
    """Cosine similarity between each image and its own caption."""
    outputs = clip_model(**inputs)
    image = F.normalize(outputs.image_embeds, dim=-1)
    text = F.normalize(outputs.text_embeds, dim=-1)
    return (image * text).sum(dim=-1)


class CLIPPairClassifier(nn.Module):
    """CLIP + a 1->2 linear head on the image-caption cosine similarity.

    Trained with cross-entropy; logit index 0 = aligned, 1 = misaligned.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(1, 2)

    def forward(self, inputs):
        return self.head(pair_similarity(self.clip_model, inputs).unsqueeze(-1))


class CLIPSimilarityModel(nn.Module):
    """CLIP used directly: 'aligned' when similarity exceeds a threshold.

    Used for the zero-shot baseline and for the contrastive fine-tune.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, inputs):
        return pair_similarity(self.clip_model, inputs)


@torch.no_grad()
def similarity_to_label(similarity, threshold):
    """Similarity above the threshold -> aligned (0), otherwise misaligned (1)."""
    return (similarity <= threshold).long()
