from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor


class CLIPDataset(Dataset):
    """
    Dataset class for loading image-text pairs.
    """

    def __init__(self, pairs, processor, max_length=77):
        self.pairs = pairs
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair["image_path"]).convert("RGB")
        inputs = self.processor(
            text=[pair["caption"]],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        label = 0 if pair["label"] == "aligned" else 1
        return {k: v.squeeze(0) for k, v in inputs.items()}, label


def collate_fn(batch):
    """
    Custom collate function for padding and batching.
    """
    inputs, labels = zip(*batch)
    max_len = max(x["input_ids"].shape[0] for x in inputs)

    padded_inputs = {
        key: torch.stack([
            torch.nn.functional.pad(
                x[key],
                pad=(0, max_len - x[key].shape[0]),
                mode="constant",
                value=0
            )
            if key in ["input_ids", "attention_mask"] else x[key]
            for x in inputs
        ])
        for key in inputs[0]
    }

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_inputs, labels
