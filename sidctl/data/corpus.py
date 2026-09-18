"""Dataset-agnostic corpus container shared by MovieLens and Amazon loaders."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from sidctl.attributes import AttributeTable

Polarity = Literal["pos", "neu", "neg"]

TextSource = Literal["title", "title_attr", "title_side", "title_side_attr"]


@dataclass
class Corpus:
    """
    A sequential-recommendation corpus with an item-attribute table.

    Item indices are contiguous in ``[0, num_items)`` and shared by
    ``item_titles``, ``item_side_text`` and ``attributes``.
    """

    name: str
    user_events: dict[int, list[dict]]
    splits: dict[str, list[int]]
    item_titles: list[str]
    item_side_text: list[str]
    attributes: AttributeTable
    meta: dict = field(default_factory=dict)

    @property
    def num_items(self) -> int:
        return len(self.item_titles)

    @property
    def num_users(self) -> int:
        return len(self.user_events)

    def item_texts(self, source: TextSource = "title_attr") -> list[str]:
        """
        Tokenizer input text.

        ``source`` is an experimental factor, not a convenience: including
        attribute names leaks the control target into the SID and inflates
        prefix controllability. ``title`` is the leakage-free condition.
        """
        out = []
        for i in range(self.num_items):
            parts = [self.item_titles[i]]
            if source in ("title_side", "title_side_attr"):
                parts.append(self.item_side_text[i])
            if source in ("title_attr", "title_side_attr"):
                parts.append(self.attributes.attribute_text(i))
            text = " ".join(p for p in parts if p).strip()
            out.append(text or f"item {i}")
        return out

    def positive_events(self, user_id: int) -> list[dict]:
        return [e for e in self.user_events[user_id] if e["polarity"] == "pos"]

    def stats(self) -> dict:
        n_events = sum(len(v) for v in self.user_events.values())
        pol = {p: 0 for p in ("pos", "neu", "neg")}
        for events in self.user_events.values():
            for e in events:
                pol[e["polarity"]] += 1
        return {
            "name": self.name,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_events": n_events,
            "events_per_user": round(n_events / max(self.num_users, 1), 2),
            "polarity_counts": pol,
            "num_attrs": self.attributes.num_attrs,
            "split_sizes": {k: len(v) for k, v in self.splits.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def label_polarity(rating: float, pos_threshold: float, neg_threshold: float) -> Polarity:
    if rating >= pos_threshold:
        return "pos"
    if rating <= neg_threshold:
        return "neg"
    return "neu"


def split_users(
    users: list[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    users = sorted(users)
    rng = np.random.default_rng(seed)
    shuffled = np.array(users)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": shuffled[:n_train].tolist(),
        "val": shuffled[n_train : n_train + n_val].tolist(),
        "test": shuffled[n_train + n_val :].tolist(),
    }


def build_attribute_matrix(
    item_attr_names: list[list[str]],
    max_attrs: int | None = None,
    min_count: int = 20,
) -> AttributeTable:
    """
    Turn per-item attribute name lists into an incidence matrix.

    Attributes appearing on fewer than ``min_count`` items are dropped; if
    ``max_attrs`` is set only the most frequent survive.
    """
    counts: dict[str, int] = {}
    for names in item_attr_names:
        for nm in set(names):
            counts[nm] = counts.get(nm, 0) + 1

    kept = [nm for nm, c in counts.items() if c >= min_count]
    kept.sort(key=lambda nm: (-counts[nm], nm))
    if max_attrs is not None:
        kept = kept[:max_attrs]

    index = {nm: a for a, nm in enumerate(kept)}
    matrix = np.zeros((len(item_attr_names), len(kept)), dtype=bool)
    for i, names in enumerate(item_attr_names):
        for nm in set(names):
            a = index.get(nm)
            if a is not None:
                matrix[i, a] = True
    return AttributeTable(names=kept, matrix=matrix)
