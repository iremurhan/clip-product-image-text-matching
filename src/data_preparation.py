"""Build aligned / misaligned image-text pairs and split them by image.

Label convention used everywhere in this repository:
    0 = aligned    (the product image with its own title)
    1 = misaligned (the product image with the title of a product that shares
                    no category with it)
"""
import ast
import json
import os
import random

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ALIGNED, MISALIGNED = 0, 1
LABEL_NAMES = {ALIGNED: "aligned", MISALIGNED: "misaligned"}


def parse_categories(value):
    """Return a product's categories as a set of strings.

    The scraped CSV stores categories as a stringified Python list
    (e.g. "['Elektronik', 'Laptop']"). Calling set() on that string directly
    gives a set of *characters*, which is the bug this function avoids.
    """
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if str(v).strip()}
    if not isinstance(value, str) or not value.strip():
        return set()
    text = value.strip()
    if text.startswith("["):
        try:
            return {str(v).strip() for v in ast.literal_eval(text) if str(v).strip()}
        except (ValueError, SyntaxError):
            pass
    return {part.strip() for part in text.split(",") if part.strip()}


def build_pairs(df, seed=42):
    """Create one aligned and (when possible) one misaligned pair per product."""
    rng = random.Random(seed)
    records = df.to_dict("records")
    categories = [parse_categories(r["categories"]) for r in records]

    aligned, misaligned = [], []
    for i, row in enumerate(records):
        aligned.append({"image_path": row["image_path"], "caption": row["title"],
                        "label": LABEL_NAMES[ALIGNED]})
        candidates = [j for j, cats in enumerate(categories)
                      if j != i and not (cats & categories[i])]
        if candidates:
            other = records[rng.choice(candidates)]
            misaligned.append({"image_path": row["image_path"], "caption": other["title"],
                               "label": LABEL_NAMES[MISALIGNED]})
    return aligned, misaligned


def split_by_image(pairs, test_size=0.2, seed=42):
    """Train/test split in which every image lands in exactly one split.

    Each image appears in both an aligned and a misaligned pair, so a random
    pair-level split puts most test images into training as well.
    """
    groups = [p["image_path"] for p in pairs]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(pairs, groups=groups))
    return [pairs[i] for i in train_idx], [pairs[i] for i in test_idx]


def prepare_data(config):
    data_cfg = config["data"]
    seed = config.get("seed", 42)

    df = pd.read_csv(data_cfg["dataset_file"])
    df = df[df["image_path"].notnull() & df["title"].notnull()]

    aligned, misaligned = build_pairs(df, seed=seed)
    train_pairs, test_pairs = split_by_image(aligned + misaligned,
                                             test_size=data_cfg.get("test_size", 0.2), seed=seed)

    os.makedirs(os.path.dirname(data_cfg["train_pairs"]) or ".", exist_ok=True)
    for path, payload in [(data_cfg["aligned_pairs"], aligned),
                          (data_cfg["misaligned_pairs"], misaligned),
                          (data_cfg["train_pairs"], train_pairs),
                          (data_cfg["test_pairs"], test_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{len(aligned)} aligned / {len(misaligned)} misaligned pairs; "
          f"{len(train_pairs)} train / {len(test_pairs)} test")
    return data_cfg["train_pairs"], data_cfg["test_pairs"]
