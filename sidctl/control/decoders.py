"""Decoders that implement an attribute constraint in different places.

The comparison is deliberately budget-matched: unless ``beam_multiplier`` says
otherwise, every decoder gets the same beam width. That is what exposes the
real trade-off -- prefix masking spends its whole budget on allowed branches but
bans coarsely, while post-filtering bans exactly but wastes budget on items it
is about to throw away.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sidctl.control.masks import MaskCache, item_has_banned_tile
from sidctl.models.tiger import TigerGR
from sidctl.sid.tokenizer import SIDTokenizer


@dataclass(frozen=True)
class DecoderSpec:
    """Where and how the constraint is enforced."""

    name: str
    mask_level: int | None = None  # ban SID prefixes at this depth
    tau: float = 0.0  # ban prefixes whose attribute share reaches tau
    item_filter: bool = False  # drop violating items after generation
    majority: bool = False  # Pmaj: ban prefixes whose majority label is attr
    allowed_trie: bool = False  # search only SIDs of allowed items
    beam_multiplier: float = 1.0
    description: str = ""


@dataclass
class DecodeResult:
    items: list[int]
    n_returned: int
    beam_size: int
    n_candidates: int  # valid SIDs the beam produced, before item filtering


DEFAULT_DECODERS: list[DecoderSpec] = [
    DecoderSpec(
        name="unconstrained",
        description="no control; reference for accuracy and natural violation rate",
    ),
    DecoderSpec(
        name="prefix_mask_l1",
        mask_level=1,
        description="ban every level-1 prefix touching the attribute "
        "(tiled: AND — drop the item if any tile is banned)",
    ),
    DecoderSpec(
        name="prefix_mask_l2",
        mask_level=2,
        description="same ban one level deeper",
    ),
    DecoderSpec(
        name="prefix_mask_l3",
        mask_level=3,
        description="same ban at the last semantic level",
    ),
    DecoderSpec(
        name="post_filter",
        item_filter=True,
        description="exact item-level ban applied to the generated list",
    ),
    DecoderSpec(
        name="post_filter_2x",
        item_filter=True,
        beam_multiplier=2.0,
        description="post-filtering given twice the beam budget",
    ),
    DecoderSpec(
        name="prefix_l1_plus_filter",
        mask_level=1,
        item_filter=True,
        tau=0.5,
        description="share≥0.5 prefixes banned, remainder filtered",
    ),
    DecoderSpec(
        name="pmaj_l1",
        mask_level=1,
        majority=True,
        description="Pmaj: ban L1 prefixes whose unique majority attribute is forbidden",
    ),
    DecoderSpec(
        name="allowed_trie",
        allowed_trie=True,
        description="beam search on the SID trie of allowed items only (dual of P0)",
    ),
]


DECODER_BY_NAME = {s.name: s for s in DEFAULT_DECODERS}

# Locked comparison: P0 vs Pmaj vs post-filter vs allowed-item trie.
POLICY_DECODER_NAMES = (
    "unconstrained",
    "prefix_mask_l1",
    "pmaj_l1",
    "post_filter",
    "allowed_trie",
)


def select_decoders(names: list[str] | None = None) -> list[DecoderSpec]:
    if not names:
        return list(DEFAULT_DECODERS)
    missing = [n for n in names if n not in DECODER_BY_NAME]
    if missing:
        known = ", ".join(DECODER_BY_NAME)
        raise SystemExit(f"unknown decoder(s) {missing}; known: {known}")
    return [DECODER_BY_NAME[n] for n in names]


def decode(
    model: TigerGR,
    tokenizer: SIDTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    spec: DecoderSpec,
    attr: int,
    mask_cache: MaskCache,
    beam_size: int = 50,
    topk: int = 10,
) -> DecodeResult:
    """Run one decoder under one constraint and return ranked item indices."""
    beam = max(int(round(beam_size * spec.beam_multiplier)), topk)

    banned_prefixes = None
    trie = tokenizer.prefix_trie
    if spec.allowed_trie:
        trie = mask_cache.allowed_trie(attr)
    elif spec.mask_level is not None:
        if spec.majority:
            banned_prefixes = mask_cache.majority_mask(attr, spec.mask_level)
        else:
            banned_prefixes = mask_cache.prefix_mask(
                attr, spec.mask_level, spec.tau
            )
        if not banned_prefixes:
            banned_prefixes = None

    scored = model.beam_search(
        input_ids,
        attention_mask,
        trie,
        beam_size=beam,
        banned_prefixes=banned_prefixes,
        max_return=beam,
    )

    drop = mask_cache.item_mask(attr) if spec.item_filter else set()
    items: list[int] = []
    seen: set[int] = set()
    for sid, _ in scored:
        item = tokenizer.sid_to_item.get(sid)
        if item is None or item in seen:
            continue
        seen.add(item)
        if item in drop:
            continue
        # Tiled AND: a horror-comedy generated via its comedy tile is still
        # excluded when horror prefixes are banned. No-op for single-SID kinds.
        if (
            banned_prefixes
            and spec.mask_level is not None
            and tokenizer.is_tiled
            and item_has_banned_tile(
                tokenizer, item, banned_prefixes, spec.mask_level
            )
        ):
            continue
        items.append(item)
        if len(items) >= topk:
            break

    return DecodeResult(
        items=items,
        n_returned=len(items),
        beam_size=beam,
        n_candidates=len(scored),
    )
