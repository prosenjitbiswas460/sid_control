"""MovieLens-1M loader. Attributes are the 18 genre labels."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from sidctl.data.corpus import (
    Corpus,
    build_attribute_matrix,
    label_polarity,
    split_users,
)

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    extracted = data_dir / "ml-1m"
    if (extracted / "ratings.dat").exists():
        return extracted

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens-1M -> {zip_path}")
        urllib.request.urlretrieve(ML1M_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return extracted


def load_movielens(
    data_dir: str | Path = "data/ml-1m",
    pos_threshold: float = 4.0,
    neg_threshold: float = 2.0,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    attr_min_count: int = 20,
    seed: int = 42,
) -> Corpus:
    root = download_movielens(data_dir)

    ratings = pd.read_csv(
        root / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        root / "movies.dat",
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )

    # Iterative k-core so that both users and items have enough support.
    for _ in range(5):
        ic = ratings.groupby("item_id").size()
        ratings = ratings[
            ratings["item_id"].isin(ic[ic >= min_item_interactions].index)
        ]
        uc = ratings.groupby("user_id").size()
        ratings = ratings[
            ratings["user_id"].isin(uc[uc >= min_user_interactions].index)
        ]

    ratings = ratings.sort_values(["user_id", "timestamp"])
    unique_items = sorted(ratings["item_id"].unique())
    item_to_idx = {iid: i for i, iid in enumerate(unique_items)}
    ratings["item_idx"] = ratings["item_id"].map(item_to_idx)

    movies = movies[movies["item_id"].isin(item_to_idx)].copy()
    movies["item_idx"] = movies["item_id"].map(item_to_idx)
    movies = movies.sort_values("item_idx").reset_index(drop=True)

    titles = movies["title"].fillna("").tolist()
    genre_lists = [
        [g for g in str(g_str).split("|") if g and g != "(no genres listed)"]
        for g_str in movies["genres"].fillna("")
    ]
    attributes = build_attribute_matrix(
        genre_lists, max_attrs=None, min_count=attr_min_count
    )

    user_events: dict[int, list[dict]] = {}
    for uid, grp in ratings.groupby("user_id"):
        user_events[int(uid)] = [
            {
                "item_idx": int(r.item_idx),
                "rating": float(r.rating),
                "timestamp": int(r.timestamp),
                "polarity": label_polarity(
                    float(r.rating), pos_threshold, neg_threshold
                ),
            }
            for r in grp.itertuples(index=False)
        ]

    return Corpus(
        name="ml-1m",
        user_events=user_events,
        splits=split_users(list(user_events), seed=seed),
        item_titles=titles,
        item_side_text=[""] * len(titles),
        attributes=attributes,
        meta={
            "attribute_family": "genre",
            "pos_threshold": pos_threshold,
            "neg_threshold": neg_threshold,
        },
    )
