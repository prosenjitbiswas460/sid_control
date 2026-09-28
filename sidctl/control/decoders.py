"""Decoders that implement an attribute constraint in different places.

The comparison is deliberately budget-matched: unless ``beam_multiplier`` says
otherwise, every decoder gets the same beam width. That is what exposes the
real trade-off -- prefix masking spends its whole budget on allowed branches but
bans coarsely, while post-filtering bans exactly but wastes budget on items it
is about to throw away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    beam: int | None = None  # fixed beam width, overrides the run's beam size
    oracle: bool = False  # score every catalog SID exactly instead of searching
    purity_weight: float = 0.0  # PACD: prune by log p + w * log allowed share
    lookahead: int = 0  # prune by log p + log allowed child mass (M * beam cands)
    description: str = ""


@dataclass
class DecodeResult:
    items: list[int]
    n_returned: int
    beam_size: int
    n_candidates: int  # valid SIDs the beam produced, before item filtering
    scores: list[float] = field(default_factory=list)  # model log-prob per item


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


SEARCH_BEAMS = (10, 20, 50, 100)


def _search_decoders() -> list[DecoderSpec]:
    specs = [
        DecoderSpec(
            name="oracle_unconstrained",
            oracle=True,
            description="exact top-k over every catalog item (no search error)",
        ),
        DecoderSpec(
            name="oracle_allowed",
            oracle=True,
            allowed_trie=True,
            description="exact top-k over allowed items: ceiling for any exact decoder",
        ),
    ]
    for b in SEARCH_BEAMS:
        specs += [
            DecoderSpec(
                name=f"allowed_trie_b{b}",
                allowed_trie=True,
                beam=b,
                description=f"allowed-item trie, beam {b}",
            ),
            DecoderSpec(
                name=f"pacd_b{b}",
                allowed_trie=True,
                beam=b,
                purity_weight=1.0,
                description=f"allowed trie, prune by log p + log allowed share, beam {b}",
            ),
            DecoderSpec(
                name=f"lookahead_b{b}",
                allowed_trie=True,
                beam=b,
                lookahead=2,
                description=f"allowed trie, prune by log p + log allowed child mass, beam {b}",
            ),
        ]
    for w in (0.5, 2.0):
        specs.append(
            DecoderSpec(
                name=f"pacd_w{w:g}_b10",
                allowed_trie=True,
                beam=10,
                purity_weight=w,
                description=f"PACD weight {w:g}, beam 10",
            )
        )
    return specs


SEARCH_DECODERS = _search_decoders()

DECODER_BY_NAME = {s.name: s for s in DEFAULT_DECODERS + SEARCH_DECODERS}

# Locked comparison: P0 vs Pmaj vs post-filter vs allowed-item trie.
POLICY_DECODER_NAMES = (
    "unconstrained",
    "prefix_mask_l1",
    "pmaj_l1",
    "post_filter",
    "allowed_trie",
)

# Search-error study: oracle ceiling, beam sweep, PACD and lookahead pruning.
SEARCH_DECODER_NAMES = ("unconstrained",) + tuple(s.name for s in SEARCH_DECODERS)


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
    if spec.oracle:
        return _oracle_decode(model, input_ids, attention_mask, spec, attr, mask_cache, topk)

    beam = max(spec.beam or int(round(beam_size * spec.beam_multiplier)), topk)

    banned_prefixes = None
    prefix_bonus = None
    trie = tokenizer.prefix_trie
    if spec.allowed_trie:
        if spec.purity_weight:
            prefix_bonus = mask_cache.allowed_log_share(attr)
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
        prefix_bonus=prefix_bonus,
        bonus_weight=spec.purity_weight,
        lookahead=spec.lookahead,
    )

    drop = mask_cache.item_mask(attr) if spec.item_filter else set()
    items: list[int] = []
    scores: list[float] = []
    seen: set[int] = set()
    for sid, score in scored:
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
        scores.append(score)
        if len(items) >= topk:
            break

    return DecodeResult(
        items=items,
        n_returned=len(items),
        beam_size=beam,
        n_candidates=len(scored),
        scores=scores,
    )


def _oracle_decode(
    model: TigerGR,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    spec: DecoderSpec,
    attr: int,
    mask_cache: MaskCache,
    topk: int,
) -> DecodeResult:
    """Exact ranking; a tiled item scores as its best tile."""
    device = input_ids.device
    key = tuple(input_ids.flatten().tolist())
    if mask_cache._oracle_user is None or mask_cache._oracle_user[0] != key:
        item_idx, sid_tokens = mask_cache.oracle_table(model, device)
        sid_scores = model.score_sids(input_ids, attention_mask, sid_tokens)
        n_items = mask_cache.attributes.num_items
        per_item = torch.full((n_items,), float("-inf"), device=device)
        per_item = per_item.scatter_reduce(0, item_idx, sid_scores, reduce="amax")
        mask_cache._oracle_user = (key, per_item)
    per_item = mask_cache._oracle_user[1]
    if spec.allowed_trie:
        per_item = per_item.clone()
        per_item[mask_cache.forbidden_index(attr, device)] = float("-inf")
    k = min(topk, int(torch.isfinite(per_item).sum()))
    top = per_item.topk(k)
    return DecodeResult(
        items=top.indices.tolist(),
        n_returned=k,
        beam_size=0,
        n_candidates=int(per_item.numel()),
        scores=top.values.tolist(),
    )
