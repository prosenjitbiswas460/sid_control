"""Wiring tests that run in seconds on CPU with synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

from collections import defaultdict

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.analysis import (
    analyze_tokenizer,
    evaluate_attr_policies,
    prefix_purity,
    realizability_frontier,
    resolve_constraint,
)
from sidctl.attributes import AttributeTable
from sidctl.control import (
    DEFAULT_DECODERS,
    MaskCache,
    build_prefix_mask,
    decode,
    select_decoders,
)
from sidctl.control.masks import build_majority_mask, item_has_banned_tile, mask_effect
from sidctl.control.protocol import build_control_instances
from sidctl.data import GRDataset, build_vocab, collate_fn
from sidctl.data.corpus import Corpus, build_attribute_matrix, split_users
from sidctl.eval import evaluate_control
from sidctl.models import TigerGR
from sidctl.sid import SIDTokenizer, build_tokenizer

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


def _multilabel_catalog():
    """Comedy-only, horror-only, and comedy+horror — the Proposal 2 stress case."""
    n = 40
    names = (
        [["comedy"]] * n
        + [["horror"]] * n
        + [["comedy", "horror"]] * n
    )
    titles = [f"movie {i} {' '.join(names[i])}" for i in range(3 * n)]
    attributes = build_attribute_matrix(names, min_count=1)
    return titles, attributes


def _tok_kwargs():
    return dict(num_levels=2, codebook_size=8, tfidf_dim=16, embed_dim=8, random_state=0)


def test_tiled_assigns_one_tile_per_label():
    titles, attributes = _multilabel_catalog()
    tok = build_tokenizer("tiled", titles, attributes=attributes, **_tok_kwargs())
    assert tok.is_tiled
    n = 40
    for i in range(n):
        assert len(tok.iter_sids(i)) == 1
    for i in range(2 * n, 3 * n):
        assert len(tok.iter_sids(i)) == 2
    all_sids = [sid for i in range(tok.num_items) for sid in tok.iter_sids(i)]
    assert len(all_sids) == len(set(all_sids))
    assert len(tok.sid_to_item) == len(all_sids)
    purity = prefix_purity(tok, attributes, level=1)
    assert purity["purity"] == pytest.approx(1.0)


def test_tiled_removes_multilabel_floor():
    titles, attributes = _multilabel_catalog()
    kwargs = dict(attributes=attributes, **_tok_kwargs())
    category = build_tokenizer("category", titles, **kwargs)
    tiled = build_tokenizer("tiled", titles, **kwargs)
    horror = attributes.names.index("horror")
    cat_coll = realizability_frontier(category, attributes, horror, 1).collateral_at(0.0)
    tiled_coll = realizability_frontier(tiled, attributes, horror, 1).collateral_at(0.0)
    tiled_leak = realizability_frontier(tiled, attributes, horror, 1).points[0]["leakage"]
    assert cat_coll > 0.2, "single-label category codes must pay the overlap floor"
    assert tiled_coll == pytest.approx(0.0)
    assert tiled_leak == pytest.approx(0.0)


def test_sliced_uses_constraint_l0_and_within_slice_residual(corpus):
    kwargs = dict(attributes=corpus.attributes, **_tok_kwargs())
    cat = build_tokenizer("category", corpus.item_titles, **kwargs)
    sl = build_tokenizer("sliced", corpus.item_titles, **kwargs)
    assert sl.is_sliced
    assert not sl.is_tiled
    assert np.array_equal(cat.sid_table[:, 0], sl.sid_table[:, 0])
    # Same constraint cut, different retrieval tail.
    assert not np.array_equal(cat.sid_table[:, 1:-1], sl.sid_table[:, 1:-1])
    labels = corpus.attributes.dominant_labels()
    for a in np.unique(labels):
        if a < 0:
            continue
        items = np.flatnonzero(labels == a)
        assert len(set(int(sl.sid_table[i, 0]) for i in items)) == 1


def test_joint_loss_is_recommendation_plus_slice_ce(corpus, tokenizer):
    from torch.utils.data import DataLoader

    from sidctl.train import Trainer

    model, vocab = _tiny_model(tokenizer)
    ds = GRDataset(corpus, tokenizer, vocab, split="train", max_history_len=5)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    plain = Trainer(
        model, vocab, loader, constraint_loss_weight=0.0, warmup_steps=0
    )
    rec, ctrl, joint = plain._losses(batch)
    assert torch.allclose(joint, rec)
    weighted = Trainer(
        model, vocab, loader, constraint_loss_weight=2.0, warmup_steps=0
    )
    rec2, ctrl2, joint2 = weighted._losses(batch)
    assert torch.allclose(joint2, rec2 + 2.0 * ctrl2)


def test_tiled_prefix_ban_and_semantics(corpus):
    tok = build_tokenizer(
        "tiled",
        corpus.item_texts("title"),
        attributes=corpus.attributes,
        **_tok_kwargs(),
    )
    banned = build_prefix_mask(tok, corpus.attributes, attr=0, level=1, tau=0.0)
    assert banned
    for i in range(corpus.num_items):
        if corpus.attributes.has(i, 0):
            assert item_has_banned_tile(tok, i, banned, 1)
        else:
            assert not item_has_banned_tile(tok, i, banned, 1)

    vocab = build_vocab(tok)
    model = TigerGR(
        vocab_size=vocab.vocab_size,
        sid_length=tok.sid_length,
        level_sizes=tok.level_sizes,
        d_model=32,
        num_layers=1,
        num_heads=2,
        d_ff=64,
    )
    ds = GRDataset(corpus, tok, vocab, split="test", max_history_len=5)
    input_ids = torch.tensor([ds[0]["input_ids"]])
    attn = torch.ones_like(input_ids)
    out = model.beam_search(
        input_ids, attn, tok.prefix_trie, beam_size=8, banned_prefixes=banned
    )
    for sid, _ in out:
        assert sid[:1] not in banned
        item = tok.sid_to_item[sid]
        # Prefix-only generation can still surface a multi-label item via
        # another tile; AND membership is applied in decode().
        if not item_has_banned_tile(tok, item, banned, 1):
            assert not corpus.attributes.has(item, 0)

    cache = MaskCache(tok, corpus.attributes)
    spec = [s for s in DEFAULT_DECODERS if s.name == "prefix_mask_l1"][0]
    result = decode(
        model, tok, input_ids, attn, spec, attr=0,
        mask_cache=cache, beam_size=8, topk=5,
    )
    for item in result.items:
        assert not corpus.attributes.has(item, 0)


def test_tiled_save_load_roundtrip(tmp_path):
    titles, attributes = _multilabel_catalog()
    tok = build_tokenizer("tiled", titles, attributes=attributes, **_tok_kwargs())
    path = tmp_path / "tokenizer.pkl"
    tok.save(path)
    loaded = SIDTokenizer.load(path)
    assert loaded.is_tiled
    assert loaded.num_items == tok.num_items
    assert loaded.iter_sids(80) == tok.iter_sids(80)
    assert loaded.sid_to_item[tok.iter_sids(80)[1]] == 80


# ---------------------------------------------------------------------------
# Frozen-prefix policies: P0 / Pτ / Pmaj
# ---------------------------------------------------------------------------


def _codes_tokenizer(l1: list[int]) -> SIDTokenizer:
    """Hand-built L1 codes plus a collision suffix."""
    n = len(l1)
    table = np.zeros((n, 2), dtype=np.int64)
    table[:, 0] = l1
    seen: dict[int, int] = defaultdict(int)
    for i, code in enumerate(l1):
        table[i, 1] = seen[code]
        seen[code] += 1
    tok = SIDTokenizer(kind="rq", num_levels=1, codebook_size=8)
    tok.sid_table = table
    tok._build_trie()
    return tok


def test_p0_has_zero_leakage(tokenizer, corpus):
    res = evaluate_attr_policies(tokenizer, corpus.attributes, attr=0, level=1)
    assert res["p0"]["leakage"] == pytest.approx(0.0)
    assert res["p0"]["coverage"] == pytest.approx(1.0)


def test_pmaj_on_disjoint_category_is_free():
    names = [["alpha"]] * 12 + [["beta"]] * 12 + [["gamma"]] * 12
    titles = [f"item {i} {names[i][0]}" for i in range(len(names))]
    attributes = build_attribute_matrix(names, min_count=1)
    tok = build_tokenizer("category", titles, attributes=attributes, **_tok_kwargs())
    alpha = attributes.names.index("alpha")
    p0 = mask_effect(
        tok, attributes, alpha, build_prefix_mask(tok, attributes, alpha, 1, 0.0), 1
    )
    pmaj = mask_effect(
        tok, attributes, alpha, build_majority_mask(tok, attributes, alpha, 1), 1
    )
    assert p0["leakage"] == pytest.approx(0.0)
    assert pmaj["leakage"] == pytest.approx(0.0)
    assert pmaj["collateral"] == pytest.approx(0.0)
    assert p0["collateral"] == pytest.approx(0.0)


def test_pmaj_is_not_the_same_as_share_threshold():
    """Plurality but not majority: Pmaj bans, Pτ@0.5 does not."""
    names = (
        [["alpha"]] * 5
        + [["beta"]] * 4
        + [["gamma"]] * 4
        + [["beta"]] * 10
        + [["gamma"]] * 10
    )
    codes = [0] * 13 + [1] * 10 + [2] * 10
    attributes = build_attribute_matrix(names, min_count=1)
    tok = _codes_tokenizer(codes)
    alpha = attributes.names.index("alpha")
    res = evaluate_attr_policies(tok, attributes, alpha, 1)
    n_clean = 4 + 4 + 10 + 10
    assert res["p0"]["leakage"] == pytest.approx(0.0)
    assert res["pmaj"]["leakage"] == pytest.approx(0.0)
    assert res["pmaj"]["collateral"] == pytest.approx(8 / n_clean)
    assert res["ptau_05"]["leakage"] == pytest.approx(1.0)
    assert res["ptau_05"]["collateral"] == pytest.approx(0.0)


def test_majority_mask_cache(corpus, tokenizer):
    cache = MaskCache(tokenizer, corpus.attributes)
    a = cache.majority_mask(0, 1)
    b = cache.majority_mask(0, 1)
    assert a is b


def test_allowed_trie_keeps_mixed_prefixes_and_zero_leak():
    """P0 closes a mixed L1 bucket; the allowed trie keeps the clean items."""
    names = [["alpha"]] * 4 + [["beta"]] * 4 + [["alpha"]] * 4 + [["beta"]] * 4
    attributes = build_attribute_matrix(names, min_count=1)
    codes = [0] * 8 + [1] * 8
    tok = _codes_tokenizer(codes)
    alpha = attributes.names.index("alpha")
    cache = MaskCache(tok, attributes)

    assert (0,) in cache.prefix_mask(alpha, 1, 0.0)
    trie = cache.allowed_trie(alpha)
    assert 0 in trie[0][()]
    assert 1 in trie[0][()]

    for item, sid in enumerate(tok.sid_table):
        sid_t = tuple(int(v) for v in sid)
        if attributes.has(item, alpha):
            continue
        assert sid_t[0] in trie[0][()]
        assert sid_t[1] in trie[1][sid_t[:1]]


def test_allowed_trie_and_pmaj_decode(corpus, tokenizer):
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
    cache = MaskCache(tokenizer, corpus.attributes)
    specs = select_decoders(["pmaj_l1", "allowed_trie", "prefix_mask_l1"])
    for spec in specs:
        result = decode(
            model, tokenizer, input_ids, attn, spec, attr=0,
            mask_cache=cache, beam_size=8, topk=5,
        )
        if spec.name in ("allowed_trie", "prefix_mask_l1"):
            for item in result.items:
                assert not corpus.attributes.has(item, 0)


def _tiny_model(tokenizer):
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
    return model.eval(), vocab


def test_oracle_matches_exhaustive_allowed_beam(corpus, tokenizer):
    model, vocab = _tiny_model(tokenizer)
    ds = GRDataset(corpus, tokenizer, vocab, split="test", max_history_len=5)
    input_ids = torch.tensor([ds[0]["input_ids"]])
    attn = torch.ones_like(input_ids)
    cache = MaskCache(tokenizer, corpus.attributes)
    oracle, beam = select_decoders(["oracle_allowed", "allowed_trie"])
    full = len(tokenizer.sid_to_item)
    exact = decode(model, tokenizer, input_ids, attn, oracle, 0, cache, topk=5)
    wide = decode(model, tokenizer, input_ids, attn, beam, 0, cache,
                  beam_size=full, topk=5)
    assert exact.items == wide.items
    assert exact.scores == pytest.approx(wide.scores, abs=1e-4)
    assert all(not corpus.attributes.has(i, 0) for i in exact.items)


def test_allowed_log_share_is_allowed_trie_with_log_fractions(corpus, tokenizer):
    cache = MaskCache(tokenizer, corpus.attributes)
    share = cache.allowed_log_share(0)
    trie = cache.allowed_trie(0)
    for level in range(tokenizer.sid_length):
        nodes = {p + (c,) for p, codes in trie[level].items() for c in codes}
        assert set(share[level]) == nodes
        assert all(v <= 1e-12 for v in share[level].values())
    # leaves are single allowed SIDs, so their share is exactly 1
    assert all(v == pytest.approx(0.0) for v in share[-1].values())


def test_search_decoders_are_exact_and_report_overlap(corpus, tokenizer):
    model, vocab = _tiny_model(tokenizer)
    instances = build_control_instances(
        corpus, allowed_attrs=list(range(corpus.attributes.num_attrs)),
        min_attr_count=1, min_history=1, max_users=4,
    )
    specs = select_decoders([
        "unconstrained", "oracle_allowed", "allowed_trie_b10",
        "pacd_b10", "lookahead_b10", "pacd_w2_b10",
    ])
    res = evaluate_control(
        model, tokenizer, vocab, corpus, instances, specs,
        beam_size=8, topk=3, max_history_len=5, device="cpu",
    )
    m = res["per_decoder"]
    assert m["oracle_allowed"]["oracle_overlap"] == pytest.approx(1.0)
    assert m["oracle_allowed"]["missed_best"] == pytest.approx(0.0)
    for name in ("allowed_trie_b10", "pacd_b10", "lookahead_b10", "pacd_w2_b10"):
        assert m[name]["violation_rate"] == pytest.approx(0.0)
        assert 0.0 <= m[name]["oracle_overlap"] <= 1.0
        assert m[name]["latency_ms"] >= 0.0


def test_exact_pruning_is_safe_and_settles_pure_prefixes():
    """A is 100% forbidden, B is mixed. Exact pruning kills A, keeps B's allowed items."""
    names = [["alpha"]] * 8 + [["alpha"]] * 2 + [["beta"]] * 6
    codes = [0] * 8 + [1] * 8
    attributes = build_attribute_matrix(names, min_count=1)
    tok = _codes_tokenizer(codes)
    alpha = attributes.names.index("alpha")
    res = resolve_constraint(tok, attributes, alpha, theta=0.0)
    assert res.feasible_recall == pytest.approx(1.0)
    assert res.lost_items == 0
    assert res.settled_by("prune", 1) == pytest.approx(8 / 16)
    assert res.work_saved == pytest.approx(8 / 16)
    # mixed prefix 1 is not pruned at L1
    d1 = next(d for d in res.by_depth if d["depth"] == 1)
    assert d1["descend"] == 1
    assert d1["prune"] == 1


def test_aggressive_pruning_trades_recall_for_work():
    names = [["alpha"]] * 8 + [["alpha"]] * 2 + [["beta"]] * 6
    codes = [0] * 8 + [1] * 8
    attributes = build_attribute_matrix(names, min_count=1)
    tok = _codes_tokenizer(codes)
    alpha = attributes.names.index("alpha")
    exact = resolve_constraint(tok, attributes, alpha, 0.0)
    coarse = resolve_constraint(tok, attributes, alpha, 0.3)
    # prefix 1 is 6/8 allowed, 2/8 forbidden → share 0.75, not pruned at 0.3
    assert coarse.feasible_recall == pytest.approx(1.0)
    # prefix 1 share of allowed is 6/8 = 0.75; prune when allowed share <= 0.8
    killed = resolve_constraint(tok, attributes, alpha, 0.8)
    assert killed.feasible_recall == pytest.approx(0.0)
    assert killed.work_saved > exact.work_saved
    assert killed.lost_items == 6


def test_category_tokenizer_settles_the_catalog_at_depth_1():
    names = [["alpha"]] * 12 + [["beta"]] * 12 + [["gamma"]] * 12
    titles = [f"item {i} {names[i][0]}" for i in range(len(names))]
    attributes = build_attribute_matrix(names, min_count=1)
    tok = build_tokenizer("category", titles, attributes=attributes, **_tok_kwargs())
    alpha = attributes.names.index("alpha")
    res = resolve_constraint(tok, attributes, alpha, 0.0)
    assert res.feasible_recall == pytest.approx(1.0)
    assert res.settled_at_depth_1 == pytest.approx(1.0)
    assert res.work_saved == pytest.approx(1.0)


def test_pruned_trie_at_theta_zero_matches_allowed_trie(corpus, tokenizer):
    cache = MaskCache(tokenizer, corpus.attributes)
    assert cache.pruned_trie(0, 0.0) == cache.allowed_trie(0)


def test_prune_decoder_stays_exact_at_theta_zero(corpus, tokenizer):
    model, vocab = _tiny_model(tokenizer)
    ds = GRDataset(corpus, tokenizer, vocab, split="test", max_history_len=5)
    input_ids = torch.tensor([ds[0]["input_ids"]])
    attn = torch.ones_like(input_ids)
    cache = MaskCache(tokenizer, corpus.attributes)
    trie, prune = select_decoders(["allowed_trie", "prune_t0"])
    a = decode(model, tokenizer, input_ids, attn, trie, 0, cache, beam_size=8, topk=5)
    b = decode(model, tokenizer, input_ids, attn, prune, 0, cache, beam_size=8, topk=5)
    assert a.items == b.items
    for item in a.items:
        assert not corpus.attributes.has(item, 0)
