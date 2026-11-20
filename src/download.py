"""Download each product's main image from its product page.

Product pages embed schema.org metadata as JSON-LD (`<script type="application/ld+json">`);
the first `image.contentUrl` entry is the main product photo.
"""
import json
import os
import time

import pandas as pd
from tqdm import tqdm


def main_image_url(html):
    """Return the main image URL from a product page's JSON-LD block, or None."""
    from bs4 import BeautifulSoup

    for tag in BeautifulSoup(html, "html.parser").find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except json.JSONDecodeError:
            continue
        image = data.get("image") if isinstance(data, dict) else None
        if isinstance(image, dict):
            urls = image.get("contentUrl") or []
            urls = [urls] if isinstance(urls, str) else urls
        elif isinstance(image, list):
            urls = image
        else:
            urls = [image] if isinstance(image, str) else []
        if urls:
            return urls[0]
    return None


def download_images(products_csv, image_dir, out_csv, delay=0.5, timeout=10):
    """Fetch the main image of every product and write the CSV with an `image_path` column.

    Images already on disk are not downloaded again, so an interrupted run can be resumed.
    """
    import requests

    os.makedirs(image_dir, exist_ok=True)
    df = pd.read_csv(products_csv)
    paths = []
    session = requests.Session()
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Downloading images"):
        path = os.path.join(image_dir, f"{row.product_id}.jpg")
        if not os.path.exists(path):
            try:
                page = session.get(row.product_link, timeout=timeout)
                page.raise_for_status()
                url = main_image_url(page.text)
                if url is None:
                    paths.append(None)
                    continue
                image = session.get(url, timeout=timeout)
                image.raise_for_status()
                with open(path, "wb") as f:
                    f.write(image.content)
                time.sleep(delay)
            except requests.RequestException:
                paths.append(None)
                continue
        paths.append(path)
    df["image_path"] = paths
    df.to_csv(out_csv, index=False)
    return int(df["image_path"].notnull().sum())
