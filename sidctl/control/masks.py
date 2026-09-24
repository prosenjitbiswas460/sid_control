"""Turning an attribute constraint into a set of banned SID prefixes."""

from __future__ import annotations

import numpy as np

from sidctl.attributes import AttributeTable
from sidctl.sid.tokenizer import SIDTokenizer


def build_prefix_mask(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attr: int,
    level: int,
    tau: float = 0.0,
) -> set[tuple]:
    """
    Prefixes to remove from the trie in order to suppress ``attr``.

    A depth-``level`` prefix is banned when it contains at least one ``attr``
    item and the share of such items reaches ``tau``. ``tau=0`` bans every
    prefix that touches the attribute, which guarantees zero leakage and is the
    setting a system would need to actually honour the constraint.

    Tiled identifiers restrict the sweep to the attribute's own control
    channel (tiles whose first code *is* that attribute). Mixing every tile
    into one prefix map would re-create the multi-label floor: a comedy
    prefix would contain horror-comedies and get banned when the user asked
    to avoid horror.
    """
    groups = (
        tokenizer.channel_prefix_items(attr, level)
        if tokenizer.is_tiled
        else tokenizer.prefix_items(level)
    )
    is_attr = attributes.matrix[:, attr]

    banned = set()
    for prefix, items in groups.items():
        hits = int(is_attr[items].sum())
        if hits > 0 and (hits / len(items)) >= tau:
            banned.add(prefix)
    return banned


def item_has_banned_tile(
    tokenizer: SIDTokenizer,
    item: int,
    banned: set[tuple],
    level: int,
) -> bool:
    """True if any SID of ``item`` starts with a banned prefix (AND rule)."""
    if not banned:
        return False
    for sid in tokenizer.iter_sids(item):
        if sid[:level] in banned:
            return True
    return False


def mask_effect(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attr: int,
    banned: set[tuple],
    level: int,
) -> dict:
    """Catalog-level consequences of a mask, independent of any model."""
    is_attr = attributes.matrix[:, attr]
    n_attr = max(int(is_attr.sum()), 1)
    n_clean = max(int((~is_attr).sum()), 1)

    removed = np.array(
        [
            item_has_banned_tile(tokenizer, i, banned, level)
            for i in range(attributes.num_items)
        ],
        dtype=bool,
    )
    return {
        "level": level,
        "banned_prefixes": len(banned),
        "leakage": float((is_attr & ~removed).sum() / n_attr),
        "collateral": float(((~is_attr) & removed).sum() / n_clean),
        "catalog_retained": float((~removed).mean()),
        "control_semantics": "tiled_and" if tokenizer.is_tiled else "single_sid",
    }


def banned_items(attributes: AttributeTable, attr: int) -> set[int]:
    """Item-level ban set, used by the post-filtering decoder."""
    return set(attributes.items_with(attr).tolist())


class MaskCache:
    """
    Memoises masks across users.

    Constraints repeat heavily across a test set (there are only a few dozen
    attributes), so building each mask once matters for runtime.
    """

    def __init__(
        self,
        tokenizer: SIDTokenizer,
        attributes: AttributeTable,
    ):
        self.tokenizer = tokenizer
        self.attributes = attributes
        self._prefix: dict[tuple[int, int, float], set[tuple]] = {}
        self._items: dict[int, set[int]] = {}

    def prefix_mask(self, attr: int, level: int, tau: float = 0.0) -> set[tuple]:
        key = (attr, level, tau)
        if key not in self._prefix:
            self._prefix[key] = build_prefix_mask(
                self.tokenizer, self.attributes, attr, level, tau
            )
        return self._prefix[key]

    def item_mask(self, attr: int) -> set[int]:
        if attr not in self._items:
            self._items[attr] = banned_items(self.attributes, attr)
        return self._items[attr]
