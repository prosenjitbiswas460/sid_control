"""Turning an attribute constraint into a set of banned SID prefixes."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

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


def build_majority_mask(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attr: int,
    level: int,
) -> set[tuple]:
    """Ban prefixes whose unique majority attribute is ``attr``.

    Occupancy of an attribute in a prefix is how many items under that prefix
    carry it. The prefix is labelled with the unique argmax occupancy; ties
    and empty prefixes get no majority and are left unbanned. This is the
    only "remap" that can change leak or collateral: it changes which
    prefixes are banned, not the integers written on the codes.

    Tiled identifiers restrict the sweep to the attribute's control channel,
    matching :func:`build_prefix_mask`.
    """
    groups = (
        tokenizer.channel_prefix_items(attr, level)
        if tokenizer.is_tiled
        else tokenizer.prefix_items(level)
    )
    matrix = attributes.matrix
    banned: set[tuple] = set()
    for prefix, items in groups.items():
        occupancy = matrix[items].sum(axis=0)
        peak = int(occupancy.max())
        if peak <= 0:
            continue
        winners = np.flatnonzero(occupancy == peak)
        if len(winners) == 1 and int(winners[0]) == attr:
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
        self._majority: dict[tuple[int, int], set[tuple]] = {}
        self._items: dict[int, set[int]] = {}
        self._allowed_trie: dict[int, list[dict[tuple, list[int]]]] = {}
        self._pruned_trie: dict[tuple, list[dict[tuple, list[int]]]] = {}
        self._prune_stats: dict[tuple, dict] = {}
        self._totals: list[Counter] | None = None
        self._log_share: dict[int, list[dict[tuple, float]]] = {}
        self._oracle: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._forbidden: dict[tuple, torch.Tensor] = {}
        self._oracle_user: tuple | None = None

    def prefix_mask(self, attr: int, level: int, tau: float = 0.0) -> set[tuple]:
        key = (attr, level, tau)
        if key not in self._prefix:
            self._prefix[key] = build_prefix_mask(
                self.tokenizer, self.attributes, attr, level, tau
            )
        return self._prefix[key]

    def majority_mask(self, attr: int, level: int) -> set[tuple]:
        key = (attr, level)
        if key not in self._majority:
            self._majority[key] = build_majority_mask(
                self.tokenizer, self.attributes, attr, level
            )
        return self._majority[key]

    def item_mask(self, attr: int) -> set[int]:
        if attr not in self._items:
            self._items[attr] = banned_items(self.attributes, attr)
        return self._items[attr]

    def allowed_trie(self, attr: int) -> list[dict[tuple, list[int]]]:
        """Trie over items that do *not* carry ``attr``.

        Prefixes that mix allowed and forbidden items stay open. Paths that
        can only complete to forbidden items are absent. This is the dual of
        a P0 prefix ban.
        """
        if attr not in self._allowed_trie:
            drop = self.item_mask(attr)
            keep = [
                i
                for i in range(self.attributes.num_items)
                if i not in drop
            ]
            self._allowed_trie[attr] = self.tokenizer.trie_for_items(keep)
        return self._allowed_trie[attr]

    def pruned_trie(self, attr: int, theta: float) -> list[dict[tuple, list[int]]]:
        """Allowed trie after pruning subtrees whose allowed share is <= ``theta``.

        ``theta = 0`` is exact and returns the allowed trie. Larger values trade
        feasible items for a coarser, cheaper constraint resolution; forbidden
        items stay unreachable either way.
        """
        if theta <= 0.0:
            return self.allowed_trie(attr)
        key = (attr, float(theta))
        if key not in self._pruned_trie:
            from sidctl.analysis.pruning import resolve_constraint

            res = resolve_constraint(self.tokenizer, self.attributes, attr, theta)
            self._pruned_trie[key] = self.tokenizer.trie_for_rows(res.surviving_rows)
            self._prune_stats[key] = res.summary()
        return self._pruned_trie[key]

    def prune_stats(self, attr: int, theta: float) -> dict:
        return self._prune_stats.get((attr, float(theta)), {})

    def _prefix_totals(self) -> list[Counter]:
        if self._totals is None:
            L = self.tokenizer.sid_length
            self._totals = [Counter() for _ in range(L)]
            for _, sid in self.tokenizer._iter_item_sids():
                for d in range(L):
                    self._totals[d][sid[: d + 1]] += 1
        return self._totals

    def allowed_log_share(self, attr: int) -> list[dict[tuple, float]]:
        """``[level][prefix] -> log(allowed SIDs / all SIDs)`` under the prefix.

        ``level`` indexes prefixes of length ``level + 1``. Only prefixes with
        at least one allowed SID appear, i.e. exactly the allowed-trie nodes.
        This is the PACD pruning bonus: under a uniform spread of the model's
        mass over a subtree, it estimates log P(allowed | prefix).
        """
        if attr not in self._log_share:
            totals = self._prefix_totals()
            drop = self.item_mask(attr)
            L = self.tokenizer.sid_length
            allowed = [Counter() for _ in range(L)]
            for item, sid in self.tokenizer._iter_item_sids():
                if item in drop:
                    continue
                for d in range(L):
                    allowed[d][sid[: d + 1]] += 1
            self._log_share[attr] = [
                {p: math.log(n / totals[d][p]) for p, n in allowed[d].items()}
                for d in range(L)
            ]
        return self._log_share[attr]

    def oracle_table(self, model, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(item_index[N], sid_tokens[N, L])`` over every catalog SID."""
        key = str(device)
        if key not in self._oracle:
            items, rows = [], []
            for item, sid in self.tokenizer._iter_item_sids():
                items.append(item)
                rows.append(model.sid_to_token_ids(sid))
            self._oracle[key] = (
                torch.tensor(items, dtype=torch.long, device=device),
                torch.tensor(rows, dtype=torch.long, device=device),
            )
        return self._oracle[key]

    def forbidden_index(self, attr: int, device) -> torch.Tensor:
        key = (attr, str(device))
        if key not in self._forbidden:
            self._forbidden[key] = torch.tensor(
                sorted(self.item_mask(attr)), dtype=torch.long, device=device
            )
        return self._forbidden[key]
