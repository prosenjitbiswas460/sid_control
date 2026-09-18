"""Single entry point for corpus loading, so scripts stay dataset-agnostic."""

from __future__ import annotations

from sidctl.data.amazon import load_amazon
from sidctl.data.corpus import Corpus
from sidctl.data.movielens import load_movielens


def load_corpus(name: str, **kwargs) -> Corpus:
    """
    ``name`` is one of ``ml-1m``, ``amazon-beauty``, ``amazon-sports``,
    ``amazon-toys``. Unknown keyword arguments are passed to the loader.
    """
    if name in ("ml-1m", "movielens", "ml1m"):
        return load_movielens(**kwargs)
    if name.startswith("amazon-"):
        return load_amazon(domain=name.split("-", 1)[1], **kwargs)
    raise ValueError(
        f"unknown dataset {name!r}; expected 'ml-1m' or 'amazon-{{beauty,sports,toys}}'"
    )


AVAILABLE = ["ml-1m", "amazon-beauty", "amazon-sports", "amazon-toys"]
