"""Item-attribute tables: the ground truth against which SID prefixes are tested."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AttributeTable:
    """
    Multi-label item x attribute incidence matrix.

    `matrix[i, a]` is True when item index `i` carries attribute `a`.
    Attributes are the user-facing control targets ("less horror", "no
    fragrances"), so they must come from metadata, never from the SID itself.
    """

    names: list[str]
    matrix: np.ndarray  # (num_items, num_attrs) bool

    def __post_init__(self) -> None:
        if self.matrix.dtype != np.bool_:
            self.matrix = self.matrix.astype(bool)
        if self.matrix.shape[1] != len(self.names):
            raise ValueError(
                f"matrix has {self.matrix.shape[1]} columns but "
                f"{len(self.names)} attribute names"
            )

    @property
    def num_items(self) -> int:
        return self.matrix.shape[0]

    @property
    def num_attrs(self) -> int:
        return self.matrix.shape[1]

    def prevalence(self) -> np.ndarray:
        """Fraction of catalog carrying each attribute."""
        return self.matrix.mean(axis=0)

    def labels_per_item(self) -> float:
        """
        Mean number of attributes per item.

        This is the key explanatory variable for prefix controllability. At 1.0
        the attribute is a partition of the catalog and a constraint can in
        principle be expressed as a set of prefixes. Above 1.0, banning one
        label necessarily drags in the co-occurring labels of the same items,
        so exact prefix-level control becomes impossible for *any* tokenizer.
        """
        return float(self.matrix.sum(axis=1).mean())

    def cooccurrence_rate(self) -> float:
        """Fraction of items carrying more than one attribute."""
        return float((self.matrix.sum(axis=1) > 1).mean())

    def items_with(self, attr: int) -> np.ndarray:
        return np.flatnonzero(self.matrix[:, attr])

    def items_without(self, attr: int) -> np.ndarray:
        return np.flatnonzero(~self.matrix[:, attr])

    def has(self, item: int, attr: int) -> bool:
        return bool(self.matrix[item, attr])

    def attrs_of(self, item: int) -> list[int]:
        return np.flatnonzero(self.matrix[item]).tolist()

    def dominant_labels(self) -> np.ndarray:
        """
        Single-label projection for clustering metrics (NMI).

        Each item is assigned its rarest attribute, which is the most
        informative one; items with no attribute get label -1.
        """
        prev = self.prevalence()
        labels = np.full(self.num_items, -1, dtype=np.int64)
        for i in range(self.num_items):
            attrs = np.flatnonzero(self.matrix[i])
            if attrs.size:
                labels[i] = attrs[np.argmin(prev[attrs])]
        return labels

    def controllable_attrs(
        self,
        min_prevalence: float = 0.01,
        max_prevalence: float = 0.5,
    ) -> list[int]:
        """
        Attributes worth testing as control targets.

        Very rare attributes give unstable estimates; attributes covering most
        of the catalog make "avoid this" a degenerate request.
        """
        prev = self.prevalence()
        return [
            a
            for a in range(self.num_attrs)
            if min_prevalence <= prev[a] <= max_prevalence
        ]

    def attribute_text(self, item: int) -> str:
        """Attribute names as text, for tokenizer input ablations."""
        return " ".join(self.names[a] for a in self.attrs_of(item))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            matrix=self.matrix,
            names=np.array(self.names, dtype=object),
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "names": self.names,
                    "num_items": self.num_items,
                    "prevalence": self.prevalence().round(5).tolist(),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "AttributeTable":
        data = np.load(Path(path).with_suffix(".npz"), allow_pickle=True)
        return cls(names=list(data["names"]), matrix=data["matrix"])
