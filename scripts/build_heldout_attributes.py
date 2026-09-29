#!/usr/bin/env python
"""Build a held-out attribute table for an existing corpus.

Writes artifacts/<dataset>/heldout/<family>/ only. Does not touch corpus.pkl
or any tokenizer.

    python scripts/build_heldout_attributes.py --config configs/amazon_beauty.yaml
    python scripts/build_heldout_attributes.py --config configs/ml1m.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.data.heldout import build_heldout_attributes  # noqa: E402
from sidctl.utils import load_config, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument(
        "--family",
        default=None,
        help="brand (Amazon) or decade (MovieLens). Default: inferred.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    art = Path(args.artifacts) / dataset
    corpus_path = art / "corpus.pkl"
    if not corpus_path.exists():
        raise SystemExit(f"missing {corpus_path}; previous prepare_data run required")

    corpus = Corpus.load(corpus_path)
    dkw = cfg.get("dataset_kwargs", {})
    table, stats = build_heldout_attributes(
        corpus,
        family=args.family,
        data_dir=dkw.get("data_dir"),
        min_count=int(dkw.get("attr_min_count", 20)),
        max_attrs=dkw.get("max_attrs", 40),
    )
    family = stats["family"]
    out = art / "heldout" / family
    table.save(out / "attributes")
    save_json(stats, out / "stats.json")
    print(f"family={family}  attrs={table.num_attrs}  "
          f"labeled={stats.get('n_labeled')}  "
          f"controllable={stats.get('controllable')}")
    print(f"wrote {out / 'attributes.npz'}  (corpus.pkl unchanged)")


if __name__ == "__main__":
    main()
