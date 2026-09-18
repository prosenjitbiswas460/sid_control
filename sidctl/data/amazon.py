"""Amazon Reviews (2014) loader -- the standard TIGER benchmark domains.

Attributes come from product metadata: either a level of the ``categories``
taxonomy (semantic) or ``brand`` (largely non-semantic). The contrast between
the two matters: a text-derived SID has some chance of organising by category
and almost none of organising by brand.
"""

from __future__ import annotations

import ast
import gzip
import json
import pickle
import urllib.request
from pathlib import Path

import pandas as pd

from sidctl.data.corpus import (
    Corpus,
    build_attribute_matrix,
    label_polarity,
    split_users,
)

BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

DOMAINS = {
    "beauty": "Beauty",
    "sports": "Sports_and_Outdoors",
    "toys": "Toys_and_Games",
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n         -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def download_amazon(domain: str, data_dir: str | Path = "data/amazon") -> tuple[Path, Path]:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
    tag = DOMAINS[domain]
    data_dir = Path(data_dir)
    reviews = _download(
        f"{BASE_URL}/reviews_{tag}_5.json.gz", data_dir / f"reviews_{tag}_5.json.gz"
    )
    meta = _download(f"{BASE_URL}/meta_{tag}.json.gz", data_dir / f"meta_{tag}.json.gz")
    return reviews, meta


def _read_reviews(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(
                (
                    d["reviewerID"],
                    d["asin"],
                    float(d.get("overall", 0.0)),
                    int(d.get("unixReviewTime", 0)),
                )
            )
    return pd.DataFrame(rows, columns=["user_id", "asin", "rating", "timestamp"])


def _read_meta(path: Path, keep_asins: set[str], cache: Path | None = None) -> dict[str, dict]:
    """
    Parse product metadata. Lines are Python dict literals, not JSON.

    Only ``keep_asins`` are retained, which keeps memory modest on the larger
    domains. The parsed subset is cached because this pass costs a few minutes.
    """
    if cache is not None and cache.exists():
        with cache.open("rb") as f:
            cached = pickle.load(f)
        if keep_asins <= set(cached):
            return {a: cached[a] for a in keep_asins}

    out: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            asin = d.get("asin")
            if asin is None or asin not in keep_asins:
                continue
            out[asin] = {
                "title": str(d.get("title", "") or ""),
                "description": str(d.get("description", "") or ""),
                "categories": d.get("categories", []) or [],
                "brand": str(d.get("brand", "") or "").strip(),
            }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump(out, f)
    return out


def _category_names(paths: list[list[str]], depth: int) -> list[str]:
    """Category labels at a fixed taxonomy depth (0 is the domain root)."""
    names = []
    for path in paths:
        if len(path) > depth:
            names.append(str(path[depth]).strip())
    return [n for n in names if n]


def load_amazon(
    domain: str = "beauty",
    data_dir: str | Path = "data/amazon",
    attribute_family: str = "category",
    category_depth: int = 1,
    pos_threshold: float = 4.0,
    neg_threshold: float = 2.0,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    attr_min_count: int = 20,
    max_attrs: int | None = 40,
    seed: int = 42,
) -> Corpus:
    reviews_path, meta_path = download_amazon(domain, data_dir)
    data_dir = Path(data_dir)

    df = _read_reviews(reviews_path)
    meta = _read_meta(
        meta_path,
        keep_asins=set(df["asin"].unique()),
        cache=data_dir / f"meta_{DOMAINS[domain]}_subset.pkl",
    )

    # Items need usable text for the tokenizer, and an attribute for the study.
    def usable(asin: str) -> bool:
        m = meta.get(asin)
        if m is None or not (m["title"] or m["description"]):
            return False
        if attribute_family == "brand":
            return bool(m["brand"])
        return bool(_category_names(m["categories"], category_depth))

    df = df[df["asin"].map(usable)]

    for _ in range(5):
        ic = df.groupby("asin").size()
        df = df[df["asin"].isin(ic[ic >= min_item_interactions].index)]
        uc = df.groupby("user_id").size()
        df = df[df["user_id"].isin(uc[uc >= min_user_interactions].index)]

    if df.empty:
        raise RuntimeError(
            f"no interactions survived filtering for domain={domain!r}; "
            "check the metadata download"
        )

    df = df.sort_values(["user_id", "timestamp"])
    asins = sorted(df["asin"].unique())
    item_to_idx = {a: i for i, a in enumerate(asins)}
    df["item_idx"] = df["asin"].map(item_to_idx)

    titles, side, attr_names = [], [], []
    for asin in asins:
        m = meta[asin]
        titles.append(m["title"])
        side.append(m["description"][:512])
        if attribute_family == "brand":
            attr_names.append([m["brand"]])
        else:
            attr_names.append(_category_names(m["categories"], category_depth))

    attributes = build_attribute_matrix(
        attr_names, max_attrs=max_attrs, min_count=attr_min_count
    )

    user_events: dict[int, list[dict]] = {}
    for uid, grp in df.groupby("user_id"):
        user_events[uid] = [
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

    # Amazon user ids are strings; remap to ints for consistency with MovieLens.
    uid_map = {u: i for i, u in enumerate(sorted(user_events))}
    user_events = {uid_map[u]: ev for u, ev in user_events.items()}

    return Corpus(
        name=f"amazon-{domain}",
        user_events=user_events,
        splits=split_users(list(user_events), seed=seed),
        item_titles=titles,
        item_side_text=side,
        attributes=attributes,
        meta={
            "attribute_family": attribute_family,
            "category_depth": category_depth,
            "domain": domain,
            "pos_threshold": pos_threshold,
            "neg_threshold": neg_threshold,
        },
    )
