#!/usr/bin/env python
"""Load a corpus and build every Semantic ID variant declared in the config.

Artifacts land in ``artifacts/<dataset>/``:
    corpus.pkl                    the interaction data plus attribute table
    attributes.npz / .json        item x attribute incidence and prevalences
    <kind>_<text_source>/tokenizer.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.data import load_corpus  # noqa: E402
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.sid import build_tokenizer  # noqa: E402
from sidctl.utils import load_config, save_json, set_seed  # noqa: E402


def tokenizer_tag(kind: str, text_source: str) -> str:
    return f"{kind}_{text_source}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--force", action="store_true", help="rebuild the corpus cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    dataset = cfg["dataset"]
    out_dir = Path(args.artifacts) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.pkl"

    if corpus_path.exists() and not args.force:
        print(f"loading cached corpus {corpus_path}")
        corpus = Corpus.load(corpus_path)
    else:
        print(f"building corpus {dataset}")
        corpus = load_corpus(dataset, **cfg.get("dataset_kwargs", {}))
        corpus.save(corpus_path)

    stats = corpus.stats()
    print("\ncorpus:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    save_json(stats, out_dir / "corpus_stats.json")
    corpus.attributes.save(out_dir / "attributes")

    prev = corpus.attributes.prevalence()
    controllable = corpus.attributes.controllable_attrs()
    print(f"\nattributes: {corpus.attributes.num_attrs} total, "
          f"{len(controllable)} in the controllable prevalence band")
    top = sorted(
        range(corpus.attributes.num_attrs), key=lambda a: -prev[a]
    )[:10]
    for a in top:
        flag = "*" if a in controllable else " "
        print(f"  {flag} {corpus.attributes.names[a]:<30s} {prev[a]:.3f}")

    sid_cfg = cfg.get("sid", {})
    for spec in cfg["tokenizers"]:
        kind, text_source = spec["kind"], spec["text_source"]
        tag = tokenizer_tag(kind, text_source)
        tok_dir = out_dir / tag
        tok_dir.mkdir(parents=True, exist_ok=True)
        tok_path = tok_dir / "tokenizer.pkl"

        if tok_path.exists() and not args.force:
            print(f"\n[{tag}] cached, skipping")
            continue

        print(f"\n[{tag}] fitting tokenizer ...")
        texts = corpus.item_texts(text_source)
        tok = build_tokenizer(
            kind,
            texts,
            attributes=corpus.attributes,
            num_levels=sid_cfg.get("num_levels", 3),
            codebook_size=sid_cfg.get("codebook_size", 256),
            tfidf_dim=sid_cfg.get("tfidf_dim", 128),
            embed_dim=sid_cfg.get("embed_dim", 64),
            random_state=cfg.get("seed", 42),
        )
        tok.save(tok_path)
        tstats = tok.stats() | {"text_source": text_source}
        save_json(tstats, tok_dir / "tokenizer_stats.json")
        for k, v in tstats.items():
            print(f"    {k}: {v}")

    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
