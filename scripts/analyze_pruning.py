#!/usr/bin/env python
"""Constraint-aware SID pruning: search-space reduction at a given recall.

How much of the identifier space can a constraint settle with a subtree
decision instead of an item-by-item check? CPU only, no model, no retraining.

    python scripts/analyze_pruning.py --config configs/amazon_beauty.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.analysis.pruning import (  # noqa: E402
    DEFAULT_THETAS,
    analyze_pruning,
    fig_pruning,
    pruning_table,
)
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.sid import SIDTokenizer  # noqa: E402
from sidctl.utils import load_config, save_json  # noqa: E402

_KNOWN_TAG_PREFIXES = ("rq_", "category_", "tiled_", "random_")


def _discover_tags(art: Path, cfg: dict, requested: list[str] | None) -> list[str]:
    tags, seen = [], set()
    for spec in cfg.get("tokenizers", []):
        tag = f"{spec['kind']}_{spec['text_source']}"
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    if art.exists():
        for path in sorted(art.glob("*/tokenizer.pkl")):
            tag = path.parent.name
            if tag not in seen and tag.startswith(_KNOWN_TAG_PREFIXES):
                tags.append(tag)
                seen.add(tag)
    if requested:
        allow = set(requested)
        tags = [t for t in tags if t in allow]
    return tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--tokenizers", nargs="+", default=None)
    ap.add_argument("--thetas", nargs="+", type=float, default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    art = Path(args.artifacts) / dataset
    corpus_path = art / "corpus.pkl"
    if not corpus_path.exists():
        raise SystemExit(
            f"missing {corpus_path}; run "
            f"python scripts/prepare_data.py --config {args.config}"
        )

    corpus = Corpus.load(corpus_path)
    attributes = corpus.attributes
    attrs = attributes.controllable_attrs()
    if not attrs:
        raise SystemExit("no attributes in the controllable prevalence band")
    thetas = args.thetas or DEFAULT_THETAS

    print(
        f"dataset={dataset}  items={corpus.num_items}  "
        f"attrs_tested={len(attrs)}  thetas={thetas}"
    )

    runs: dict[str, dict] = {}
    for tag in _discover_tags(art, cfg, args.tokenizers):
        tok_path = art / tag / "tokenizer.pkl"
        if not tok_path.exists():
            print(f"[{tag}] missing tokenizer, skip")
            continue
        try:
            tok = SIDTokenizer.load(tok_path)
        except (ValueError, KeyError, OSError) as exc:
            print(f"[{tag}] cannot load tokenizer ({exc}); skip")
            continue
        print(f"[{tag}] resolving constraints ...")
        runs[tag] = analyze_pruning(tok, attributes, attrs=attrs, thetas=thetas)
        e = runs[tag]["aggregate_by_theta"][0]
        print(
            f"  exact: visits={e['node_visits']:.0f} "
            f"work_saved={e['work_saved']:.3f} "
            f"settled@d1={e['settled_at_depth_1']:.3f}"
        )

    if not runs:
        raise SystemExit("no tokenizers analysed")

    table = pruning_table(runs)
    print("\n" + "\n".join(table))

    out = Path(args.results) / dataset
    save_json(
        {"dataset": dataset, "thetas": thetas,
         "attrs_tested": [attributes.names[a] for a in attrs], "runs": runs},
        out / "pruning.json",
    )
    header = (
        f"# Constraint-aware SID pruning: {dataset}\n\n"
        "Work is node visits to resolve one constraint over the SID trie; the\n"
        "baseline is one check per SID row. `theta = 0` is exact pruning and is\n"
        "equivalent to the allowed-item trie.\n\n"
    )
    (out / "pruning.md").write_text(header + "\n".join(table))
    print(f"wrote {out / 'pruning.json'}")
    print(f"wrote {out / 'pruning.md'}")

    if not args.no_figure:
        fig_dir = Path(args.figures) / dataset
        fig_dir.mkdir(parents=True, exist_ok=True)
        path = fig_dir / "fig6_pruning.png"
        fig_pruning(runs, path, dataset=dataset)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
