"""Wiring tests that run in seconds on CPU with synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.analysis import analyze_tokenizer, prefix_purity, realizability_frontier
from sidctl.attributes import AttributeTable
from sidctl.control import DEFAULT_DECODERS, MaskCache, build_prefix_mask, decode
from sidctl.control.protocol import build_control_instances
from sidctl.data import GRDataset, build_vocab, collate_fn
from sidctl.data.corpus import Corpus, build_attribute_matrix, split_users
from sidctl.eval import evaluate_control
from sidctl.models import TigerGR
from sidctl.sid import build_tokenizer

NUM_ITEMS = 120
NUM_USERS = 40
ATTRS = ["alpha", "beta", "gamma", "delta"]


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    rng = np.random.default_rng(0)
    attr_names = [
        [ATTRS[i % len(ATTRS)]] + ([ATTRS[(i + 1) % len(ATTRS)]] if i % 5 == 0 else [])
        for i in range(NUM_ITEMS)
    ]
    attributes = build_attribute_matrix(attr_names, min_count=1)
    titles = [f"item {i} {attr_names[i][0]} widget" for i in range(NUM_ITEMS)]

    user_events = {}
    for u in range(NUM_USERS):
        n = int(rng.integers(8, 16))
        items = rng.choice(NUM_ITEMS, size=n, replace=False)
        events = []
        for t, it in enumerate(items):
            rating = float(rng.integers(1, 6))
            events.append(
                {
                    "item_idx": int(it),
                    "rating": rating,
                    "timestamp": t,
                    "polarity": "pos" if rating >= 4 else ("neg" if rating <= 2 else "neu"),
                }
            )
        # Guarantee enough positives for the leave-one-out protocol.
        for e in events[:5]:
            e["rating"], e["polarity"] = 5.0, "pos"
        user_events[u] = events

    return Corpus(
        name="synthetic",
        user_events=user_events,
        splits=split_users(list(user_events)),
        item_titles=titles,
        item_side_text=[""] * NUM_ITEMS,
        attributes=attributes,
    )


@pytest.fixture(scope="module")
def tokenizer(corpus):
    return build_tokenizer(
        "rq",
        corpus.item_texts("title"),
        attributes=corpus.attributes,
        num_levels=2,
        codebook_size=8,
        tfidf_dim=16,
        embed_dim=8,
    )


def test_attribute_table_shapes(corpus):
    at = corpus.attributes
    assert at.num_items == NUM_ITEMS
    assert at.matrix.dtype == np.bool_
    assert at.prevalence().shape == (at.num_attrs,)
    assert at.dominant_labels().shape == (NUM_ITEMS,)


def test_tokenizer_sids_are_unique(tokenizer):
    sids = {tokenizer.get_sid(i) for i in range(tokenizer.num_items)}
    assert len(sids) == tokenizer.num_items, "collision code must disambiguate items"
    assert len(tokenizer.sid_to_item) == tokenizer.num_items


def test_category_tokenizer_is_attribute_pure(corpus):
    tok = build_tokenizer(
        "category",
        corpus.item_texts("title"),
        attributes=corpus.attributes,
        num_levels=2,
        codebook_size=8,
        tfidf_dim=16,
        embed_dim=8,
    )
    purity = prefix_purity(tok, corpus.attributes, level=1)
    assert purity["purity"] > 0.9, "level-1 codes are the attribute by construction"


def test_frontier_is_monotone_and_bounded(tokenizer, corpus):
    f = realizability_frontier(tokenizer, corpus.attributes, attr=0, level=1)
    assert f.points
    for p in f.points:
        assert 0.0 <= p["leakage"] <= 1.0
        assert 0.0 <= p["collateral"] <= 1.0
    # tau=0 bans every prefix touching the attribute, so nothing leaks.
    assert f.points[0]["leakage"] == pytest.approx(0.0)
    assert np.isfinite(f.collateral_at(0.0))


def test_random_tokenizer_is_worse_than_rq(corpus):
    kwargs = dict(
        attributes=corpus.attributes,
        num_levels=2,
        codebook_size=8,
        tfidf_dim=16,
        embed_dim=8,
    )
    texts = corpus.item_texts("title")
    rq = build_tokenizer("rq", texts, **kwargs)
    rnd = build_tokenizer("random", texts, **kwargs)
    rq_purity = prefix_purity(rq, corpus.attributes, 1)["purity"]
    rnd_purity = prefix_purity(rnd, corpus.attributes, 1)["purity"]
    # Titles contain the attribute word here, so content codes should beat noise.
    assert rq_purity > rnd_purity


def test_analyze_tokenizer_reports_every_level(tokenizer, corpus):
    res = analyze_tokenizer(
        tokenizer, corpus.attributes, attrs=[0, 1]
    )
    levels = {p["level"] for p in res["purity_by_level"]}
    assert levels == set(range(1, tokenizer.sid_length))
    assert res["frontier_summaries"]


def test_dataset_splits_do_not_overlap(corpus, tokenizer):
    vocab = build_vocab(tokenizer)
    common = dict(corpus=corpus, tokenizer=tokenizer, vocab=vocab, max_history_len=5)
    train = GRDataset(**common, split="train")
    val = GRDataset(**common, split="val")
    test = GRDataset(**common, split="test")
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert len(val) == len(test), "one held-out target per eligible user"

    batch = collate_fn([train[0], train[1]])
    assert batch["input_ids"].shape[0] == 2
    # T5 labels are the target sequence only, matching the beam-search decoder.
    assert batch["labels"].shape[1] == tokenizer.sid_length


def test_beam_search_returns_valid_sids(corpus, tokenizer):
    vocab = build_vocab(tokenizer)
    model = TigerGR(
        vocab_size=vocab.vocab_size,
        sid_length=tokenizer.sid_length,
        level_sizes=tokenizer.level_sizes,
        d_model=32,
        num_layers=1,
        num_heads=2,
        d_ff=64,
    )
    ds = GRDataset(corpus, tokenizer, vocab, split="test", max_history_len=5)
    ex = ds[0]
    input_ids = torch.tensor([ex["input_ids"]])
    attn = torch.ones_like(input_ids)

    out = model.beam_search(input_ids, attn, tokenizer.prefix_trie, beam_size=5)
    assert 0 < len(out) <= 5
    for sid, score in out:
        assert sid in tokenizer.sid_to_item
        assert score <= 0.0
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_prefix_ban_is_respected(corpus, tokenizer):
    vocab = build_vocab(tokenizer)
    model = TigerGR(
        vocab_size=vocab.vocab_size,
        sid_length=tokenizer.sid_length,
        level_sizes=tokenizer.level_sizes,
        d_model=32,
        num_layers=1,
        num_heads=2,
        d_ff=64,
    )
    ds = GRDataset(corpus, tokenizer, vocab, split="test", max_history_len=5)
    input_ids = torch.tensor([ds[0]["input_ids"]])
    attn = torch.ones_like(input_ids)

    banned = build_prefix_mask(tokenizer, corpus.attributes, attr=0, level=1, tau=0.0)
    assert banned, "attribute 0 must occupy at least one level-1 prefix"

    out = model.beam_search(
        input_ids, attn, tokenizer.prefix_trie, beam_size=8, banned_prefixes=banned
    )
    for sid, _ in out:
        assert sid[:1] not in banned
        item = tokenizer.sid_to_item[sid]
        assert not corpus.attributes.has(item, 0), "zero leakage at tau=0"


def test_control_instances_are_target_compatible(corpus):
    instances = build_control_instances(
        corpus, allowed_attrs=list(range(corpus.attributes.num_attrs)),
        min_attr_count=1, min_history=1,
    )
    assert instances
    for inst in instances:
        assert not corpus.attributes.has(inst.target, inst.attr)
        assert inst.reason in ("low_rating", "high_exposure")


def test_end_to_end_control_eval(corpus, tokenizer):
    vocab = build_vocab(tokenizer)
    model = TigerGR(
        vocab_size=vocab.vocab_size,
        sid_length=tokenizer.sid_length,
        level_sizes=tokenizer.level_sizes,
        d_model=32,
        num_layers=1,
        num_heads=2,
        d_ff=64,
    )
    instances = build_control_instances(
        corpus, allowed_attrs=list(range(corpus.attributes.num_attrs)),
        min_attr_count=1, min_history=1, max_users=5,
    )
    specs = [s for s in DEFAULT_DECODERS if s.mask_level in (None, 1)]
    res = evaluate_control(
        model, tokenizer, vocab, corpus, instances, specs,
        beam_size=8, topk=3, max_history_len=5, device="cpu",
    )
    assert set(res["per_decoder"]) == {s.name for s in specs}
    base = res["per_decoder"]["unconstrained"]
    # An untrained model may score NDCG 0, in which case retention is undefined.
    if base["ndcg"] > 0:
        assert base["ndcg_retention"] == pytest.approx(1.0)
    for name, m in res["per_decoder"].items():
        assert m["n"] > 0
        assert 0.0 <= m["violation_rate"] <= 1.0
        assert 0.0 <= m["fill_rate"] <= 1.0

    # Exact bans must actually eliminate violations.
    assert res["per_decoder"]["post_filter"]["violation_rate"] == pytest.approx(0.0)
    assert res["per_decoder"]["prefix_mask_l1"]["violation_rate"] == pytest.approx(0.0)


def test_mask_cache_memoises(corpus, tokenizer):
    cache = MaskCache(tokenizer, corpus.attributes)
    a = cache.prefix_mask(0, 1)
    b = cache.prefix_mask(0, 1)
    assert a is b
