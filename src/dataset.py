import json

from PIL import Image
from torch.utils.data import Dataset

from src.data_preparation import ALIGNED, MISALIGNED


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class CLIPPairDataset(Dataset):
    """Image-caption pairs, returned as CLIP processor tensors plus a label."""

    def __init__(self, pairs, processor, max_length=77):
        self.pairs = pairs
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair["image_path"]).convert("RGB")
        inputs = self.processor(text=[pair["caption"]], images=[image], return_tensors="pt",
                                truncation=True, max_length=self.max_length,
                                padding="max_length")
        label = ALIGNED if pair["label"] == "aligned" else MISALIGNED
        return {k: v.squeeze(0) for k, v in inputs.items()}, label
