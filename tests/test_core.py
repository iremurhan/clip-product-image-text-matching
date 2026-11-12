"""Fast tests that need no images, no network and no CLIP weights."""
import pandas as pd
import torch

from src.data_preparation import (ALIGNED, MISALIGNED, build_pairs, parse_categories,
                                  split_by_image)
from src.losses import contrastive_pair_loss
from src.models import similarity_to_label


def test_parse_categories_handles_stringified_list():
    assert parse_categories("['Elektronik', 'Laptop']") == {"Elektronik", "Laptop"}
    assert parse_categories(["A", "B"]) == {"A", "B"}
    assert parse_categories(None) == set()


def test_contrastive_loss_rewards_correct_direction():
    labels = torch.tensor([ALIGNED, MISALIGNED])
    good = contrastive_pair_loss(torch.tensor([0.95, 0.05]), labels)
    bad = contrastive_pair_loss(torch.tensor([0.05, 0.95]), labels)
    assert good < bad


def test_similarity_threshold_maps_high_similarity_to_aligned():
    preds = similarity_to_label(torch.tensor([0.9, 0.1]), threshold=0.5)
    assert preds.tolist() == [ALIGNED, MISALIGNED]


def _toy_df(n=40):
    cats = [["Electronics"], ["Clothing"], ["Kitchen"], ["Books"]]
    return pd.DataFrame({
        "image_path": [f"img_{i}.jpg" for i in range(n)],
        "title": [f"product {i}" for i in range(n)],
        "categories": [str(cats[i % 4]) for i in range(n)],
    })


def test_misaligned_pairs_never_share_a_category():
    df = _toy_df()
    _, misaligned = build_pairs(df)
    title_to_cat = dict(zip(df["title"], df["categories"]))
    image_to_cat = dict(zip(df["image_path"], df["categories"]))
    assert len(misaligned) == len(df)
    for p in misaligned:
        assert title_to_cat[p["caption"]] != image_to_cat[p["image_path"]]


def test_split_is_image_disjoint():
    aligned, misaligned = build_pairs(_toy_df())
    train, test = split_by_image(aligned + misaligned, test_size=0.25)
    assert {p["image_path"] for p in train}.isdisjoint({p["image_path"] for p in test})
    assert len(train) + len(test) == len(aligned) + len(misaligned)
