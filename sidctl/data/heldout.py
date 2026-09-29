"""Second attribute family on a frozen catalog, without rebuilding SIDs.

In-family labels (genre / Amazon category) already sit on ``Corpus.attributes``.
This module attaches a *held-out* family to the same item indices:

- Amazon: product ``brand``, recovered from 2014 metadata by matching title
  and side text of the existing corpus.
- MovieLens: decade bins parsed from the ``(YYYY)`` suffix in the title.

Nothing here writes ``corpus.pkl`` or tokenizer pickles.
"""

from __future__ import annotations

import re
from pathlib import Path

from sidctl.attributes import AttributeTable
from sidctl.data.amazon import DOMAINS, _read_meta, _read_reviews, download_amazon
from sidctl.data.corpus import Corpus, build_attribute_matrix

_YEAR = re.compile(r"\((\d{4})\)\s*$")


def decade_label(title: str) -> str | None:
    """``1995`` in ``Toy Story (1995)`` → ``1990s``; None if unparseable."""
    m = _YEAR.search(title or "")
    if not m:
        return None
    year = int(m.group(1))
    if year < 1870 or year > 2030:
        return None
    return f"{(year // 10) * 10}s"


def _amazon_domain(corpus: Corpus) -> str:
    if corpus.meta.get("domain"):
        return str(corpus.meta["domain"])
    if corpus.name.startswith("amazon-"):
        return corpus.name.split("-", 1)[1]
    raise ValueError(f"cannot infer Amazon domain from {corpus.name!r}")


def amazon_brands_for_corpus(
    corpus: Corpus,
    data_dir: str | Path = "data/amazon",
) -> tuple[list[list[str]], dict]:
    """Per-item brand name lists aligned to ``corpus`` item indices."""
    domain = _amazon_domain(corpus)
    data_dir = Path(data_dir)
    reviews_path, meta_path = download_amazon(domain, data_dir)
    df = _read_reviews(reviews_path)
    meta = _read_meta(
        meta_path,
        keep_asins=set(df["asin"].unique()),
        cache=data_dir / f"meta_{DOMAINS[domain]}_subset.pkl",
    )
    by_both: dict[tuple[str, str], list[str]] = {}
    by_title: dict[str, list[str]] = {}
    for asin, m in meta.items():
        title = m["title"]
        side = m["description"][:512]
        by_both.setdefault((title, side), []).append(asin)
        by_title.setdefault(title, []).append(asin)

    names: list[list[str]] = []
    matched, ambiguous = 0, 0
    for i, title in enumerate(corpus.item_titles):
        side = corpus.item_side_text[i]
        asins = by_both.get((title, side)) or []
        if len(asins) != 1:
            cands = by_title.get(title, [])
            asins = cands if len(cands) == 1 else []
            if len(cands) > 1:
                ambiguous += 1
        if len(asins) != 1:
            names.append([])
            continue
        matched += 1
        brand = str(meta[asins[0]].get("brand") or "").strip()
        names.append([brand] if brand else [])
    stats = {
        "family": "brand",
        "n_items": corpus.num_items,
        "matched": matched,
        "ambiguous_titles": ambiguous,
        "with_brand": sum(1 for n in names if n),
    }
    return names, stats


def movielens_decades_for_corpus(corpus: Corpus) -> tuple[list[list[str]], dict]:
    names = []
    parsed = 0
    for title in corpus.item_titles:
        lab = decade_label(title)
        if lab:
            parsed += 1
            names.append([lab])
        else:
            names.append([])
    stats = {
        "family": "decade",
        "n_items": corpus.num_items,
        "parsed": parsed,
    }
    return names, stats


def build_heldout_attributes(
    corpus: Corpus,
    family: str | None = None,
    data_dir: str | Path | None = None,
    min_count: int = 20,
    max_attrs: int | None = 40,
) -> tuple[AttributeTable, dict]:
    """Held-out labels for this corpus. Does not modify ``corpus``."""
    if family is None:
        family = "decade" if corpus.name in ("ml-1m",) else "brand"
    if family == "brand":
        raw, stats = amazon_brands_for_corpus(
            corpus, data_dir or "data/amazon"
        )
    elif family == "decade":
        raw, stats = movielens_decades_for_corpus(corpus)
    else:
        raise ValueError(f"unknown held-out family {family!r}")
    table = build_attribute_matrix(raw, max_attrs=max_attrs, min_count=min_count)
    stats["n_attrs"] = table.num_attrs
    stats["n_labeled"] = int(table.matrix.any(axis=1).sum())
    stats["controllable"] = len(table.controllable_attrs())
    return table, stats
