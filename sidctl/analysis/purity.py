"""Prefix purity and the attribute-control realisability frontier.

This module answers the paper's first question without touching a GPU: given a
Semantic ID table and an item-attribute table, *can* an attribute constraint be
expressed as a set of banned SID prefixes, and what does it cost?

Two quantities matter, both defined per attribute ``a`` and prefix depth ``l``:

leakage
    Fraction of ``a``-items that survive the mask. The user asked not to see
    these, so leakage is the constraint violation.
collateral
    Fraction of non-``a`` items destroyed by the mask. These items were
    perfectly acceptable, so collateral is the price of using prefixes as the
    control surface.

A perfect control surface reaches zero leakage at zero collateral. The headline
measurement of the paper is how far a real RQ tokenizer is from that corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from sidctl.attributes import AttributeTable
from sidctl.sid.tokenizer import SIDTokenizer

# np.trapezoid replaced np.trapz in NumPy 2.0.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

DEFAULT_THRESHOLDS = [
    0.0,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


def _prefix_index(tokenizer: SIDTokenizer, level: int) -> tuple[np.ndarray, int]:
    """Dense prefix id per item, plus the number of distinct prefixes."""
    prefixes = tokenizer.item_prefixes(level)
    _, inverse = np.unique(prefixes, axis=0, return_inverse=True)
    ids = np.asarray(inverse).reshape(-1).astype(np.int64)
    return ids, int(ids.max()) + 1


def prefix_purity(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    level: int,
) -> dict:
    """
    How well does depth-``level`` prefix membership predict attributes?

    ``purity`` is the size-weighted mean, over prefixes, of the largest
    attribute share inside that prefix. ``nmi`` compares the prefix partition
    to a single-label projection of the attributes.
    """
    ids, num_prefixes = _prefix_index(tokenizer, level)
    counts = np.bincount(ids, minlength=num_prefixes).astype(np.float64)

    # (num_prefixes, num_attrs) attribute counts per prefix.
    per_attr = np.zeros((num_prefixes, attributes.num_attrs), dtype=np.float64)
    for a in range(attributes.num_attrs):
        per_attr[:, a] = np.bincount(
            ids, weights=attributes.matrix[:, a].astype(np.float64),
            minlength=num_prefixes,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(counts[:, None] > 0, per_attr / counts[:, None], 0.0)
    purity = float((shares.max(axis=1) * counts).sum() / max(counts.sum(), 1.0))

    labels = attributes.dominant_labels()
    keep = labels >= 0
    nmi = (
        float(normalized_mutual_info_score(labels[keep], ids[keep]))
        if keep.any()
        else float("nan")
    )

    # Spread: fraction of prefixes an attribute touches. Near 1.0 means the
    # attribute is scattered across the whole tree and cannot be masked away.
    touched = (per_attr > 0).sum(axis=0) / max(num_prefixes, 1)

    return {
        "level": level,
        "num_prefixes": num_prefixes,
        "mean_prefix_size": float(counts.mean()),
        "purity": purity,
        "nmi": nmi,
        "mean_attr_spread": float(touched.mean()),
        "attr_spread": touched.tolist(),
    }


@dataclass
class Frontier:
    """Leakage/collateral trade-off for one attribute at one prefix depth."""

    attr: int
    attr_name: str
    level: int
    prevalence: float
    points: list[dict] = field(default_factory=list)

    def collateral_at(self, max_leakage: float) -> float:
        """Smallest collateral achievable without exceeding ``max_leakage``."""
        ok = [p["collateral"] for p in self.points if p["leakage"] <= max_leakage]
        return float(min(ok)) if ok else float("nan")

    def summary(self) -> dict:
        return {
            "attr": self.attr,
            "attr_name": self.attr_name,
            "level": self.level,
            "prevalence": self.prevalence,
            "collateral_at_zero_leakage": self.collateral_at(0.0),
            "collateral_at_leakage_01": self.collateral_at(0.01),
            "collateral_at_leakage_05": self.collateral_at(0.05),
            "aulc": self.area(),
        }

    def area(self) -> float:
        """Area under the leakage-collateral curve; 0 is a perfect control."""
        pts = sorted(
            ((p["leakage"], p["collateral"]) for p in self.points), key=lambda x: x[0]
        )
        if len(pts) < 2:
            return float("nan")
        xs, ys = zip(*pts)
        return float(_trapezoid(ys, xs)) if xs[-1] > xs[0] else float("nan")


def realizability_frontier(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attr: int,
    level: int,
    thresholds: list[float] | None = None,
) -> Frontier:
    """
    Sweep the masking threshold and record (leakage, collateral).

    A prefix is banned when it contains at least one ``attr`` item *and* the
    share of ``attr`` items inside it is at least ``tau``. Small ``tau`` bans
    aggressively (no leakage, much collateral); ``tau`` near 1 bans only
    attribute-pure prefixes (little collateral, much leakage).
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    ids, num_prefixes = _prefix_index(tokenizer, level)

    is_attr = attributes.matrix[:, attr]
    counts = np.bincount(ids, minlength=num_prefixes).astype(np.float64)
    attr_counts = np.bincount(
        ids, weights=is_attr.astype(np.float64), minlength=num_prefixes
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(counts > 0, attr_counts / counts, 0.0)

    n_attr = float(is_attr.sum())
    n_clean = float((~is_attr).sum())

    frontier = Frontier(
        attr=attr,
        attr_name=attributes.names[attr],
        level=level,
        prevalence=float(n_attr / max(len(is_attr), 1)),
    )
    if n_attr == 0 or n_clean == 0:
        return frontier

    for tau in thresholds:
        banned = (attr_counts > 0) & (shares >= tau)
        removed_item = banned[ids]

        leakage = float((is_attr & ~removed_item).sum() / n_attr)
        collateral = float(((~is_attr) & removed_item).sum() / n_clean)
        frontier.points.append(
            {
                "tau": float(tau),
                "leakage": leakage,
                "collateral": collateral,
                "banned_prefixes": int(banned.sum()),
                "prefix_ban_rate": float(banned.sum() / max(num_prefixes, 1)),
                "catalog_retained": float((~removed_item).mean()),
            }
        )
    return frontier


def analyze_tokenizer(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    levels: list[int] | None = None,
    attrs: list[int] | None = None,
    thresholds: list[float] | None = None,
) -> dict:
    """Full Part-A analysis for one tokenizer: purity plus frontiers."""
    levels = levels or list(range(1, tokenizer.sid_length))
    attrs = attrs if attrs is not None else attributes.controllable_attrs()

    purity = [prefix_purity(tokenizer, attributes, l) for l in levels]

    frontiers: list[Frontier] = []
    for level in levels:
        for a in attrs:
            frontiers.append(
                realizability_frontier(
                    tokenizer, attributes, a, level, thresholds=thresholds
                )
            )

    summaries = [f.summary() for f in frontiers]
    by_level = {}
    for level in levels:
        rows = [s for s in summaries if s["level"] == level]
        finite = [
            r["collateral_at_zero_leakage"]
            for r in rows
            if np.isfinite(r["collateral_at_zero_leakage"])
        ]
        by_level[level] = {
            "mean_collateral_at_zero_leakage": float(np.mean(finite)) if finite else float("nan"),
            "median_collateral_at_zero_leakage": float(np.median(finite)) if finite else float("nan"),
            "mean_aulc": float(
                np.nanmean([r["aulc"] for r in rows])
            ) if rows else float("nan"),
        }

    return {
        "tokenizer": tokenizer.stats(),
        "attrs_tested": [attributes.names[a] for a in attrs],
        "purity_by_level": purity,
        "frontier_summaries": summaries,
        "frontier_points": [
            {"attr_name": f.attr_name, "level": f.level, "points": f.points}
            for f in frontiers
        ],
        "aggregate_by_level": by_level,
    }
