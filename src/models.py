import torch.nn as nn
from transformers import CLIPModel


class CLIPClassifier(nn.Module):
    """
    Binary classifier for aligned vs misaligned using CLIP embeddings.
    """

    def __init__(self, clip_model):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(1, 2)

    def forward(self, inputs):
        outputs = self.clip_model(**inputs)
        similarity = torch.einsum("ij,ij->i", outputs.image_embeds, outputs.text_embeds).unsqueeze(-1)
        logits = self.fc(similarity)
        return logits