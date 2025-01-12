import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(config):
    """
    Prepares aligned and misaligned pairs and splits them into training and testing datasets.
    """
    dataset_file = config["data"]["dataset_file"]
    aligned_path = config["data"]["aligned_pairs"]
    misaligned_path = config["data"]["misaligned_pairs"]
    train_path = config["data"]["train_pairs"]
    test_path = config["data"]["test_pairs"]

    df = pd.read_csv(dataset_file)
    df = df[df["image_path"].notnull()]

    # Create aligned and misaligned pairs
    aligned_pairs, misaligned_pairs = [], []
    for _, row in df.iterrows():
        aligned_pairs.append({"image_path": row["image_path"], "caption": row["title"], "label": "aligned"})

        unrelated_rows = df[df["categories"].apply(lambda x: not set(x).intersection(row["categories"]))]
        if not unrelated_rows.empty:
            random_row = unrelated_rows.sample(1).iloc[0]
            misaligned_pairs.append({
                "image_path": row["image_path"],
                "caption": random_row["title"],
                "label": "misaligned"
            })

    # Save aligned/misaligned pairs
    with open(aligned_path, "w") as f:
        json.dump(aligned_pairs, f, indent=4)

    with open(misaligned_path, "w") as f:
        json.dump(misaligned_pairs, f, indent=4)

    # Combine and split train/test pairs
    all_pairs = aligned_pairs + misaligned_pairs
    train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)

    with open(train_path, "w") as f:
        json.dump(train_pairs, f, indent=4)

    with open(test_path, "w") as f:
        json.dump(test_pairs, f, indent=4)