#!/usr/bin/env python
"""Same frozen SIDs, in-family constraint vs a held-out family.

CPU only. Reads existing tokenizers and corpus.pkl. Writes
results/<dataset>/heldout.json and figures/<dataset>/fig7_heldout.png.
Does not overwrite control_eval.json, pruning.json, or checkpoints.

    python scripts/eval_heldout_constraint.py --config configs/amazon_beauty.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.analysis.heldout import fig_heldout, heldout_table, measure_family  # noqa: E402
from sidctl.attributes import AttributeTable  # noqa: E402
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.sid import SIDTokenizer  # noqa: E402
from sidctl.utils import load_config, save_json  # noqa: E402

_KNOWN_TAG_PREFIXES = ("rq_", "category_", "tiled_", "sliced_", "random_")


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


def _drop_heavy(block: dict) -> dict:
    keep = {
        k: v
        for k, v in block.items()
        if k not in ("policies", "pruning", "attr_names")
    }
    keep["n_attrs"] = block.get("n_attrs")
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--tokenizers", nargs="+", default=None)
    ap.add_argument("--family", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    art = Path(args.artifacts) / dataset
    corpus = Corpus.load(art / "corpus.pkl")
    in_family = corpus.attributes
    family = args.family or ("decade" if dataset == "ml-1m" else "brand")
    held_path = art / "heldout" / family / "attributes.npz"
    if not held_path.exists():
        raise SystemExit(
            f"missing {held_path}; run "
            f"python scripts/build_heldout_attributes.py --config {args.config}"
        )
    held = AttributeTable.load(art / "heldout" / family / "attributes")
    if held.num_items != corpus.num_items:
        raise SystemExit(
            f"held-out table has {held.num_items} items, corpus has {corpus.num_items}"
        )

    print(
        f"dataset={dataset}  items={corpus.num_items}  "
        f"in_family={in_family.num_attrs} attrs  "
        f"held_out={family} {held.num_attrs} attrs"
    )

    runs: dict[str, dict] = {}
    for tag in _discover_tags(art, cfg, args.tokenizers):
        tok_path = art / tag / "tokenizer.pkl"
        if not tok_path.exists():
            print(f"[{tag}] missing tokenizer, skip")
            continue
        tok = SIDTokenizer.load(tok_path)
        print(f"[{tag}] in-family ...")
        inn = measure_family(tok, in_family, ignore_tile_channels=False)
        print(f"[{tag}] held-out {family} ...")
        hld = measure_family(tok, held, ignore_tile_channels=True)
        runs[tag] = {
            "in_family": inn,
            "held_out": hld,
            "in_family_name": corpus.meta.get("attribute_family", "in_family"),
            "held_out_name": family,
        }
        print(
            f"  coll L1  in={inn['mean_collateral_l1']:.3f}  "
            f"held={hld['mean_collateral_l1']:.3f}  "
            f"d1 in={inn['settled_at_depth_1']:.3f}  "
            f"held={hld['settled_at_depth_1']:.3f}"
        )

    if not runs:
        raise SystemExit("no tokenizers evaluated")

    slim = {
        tag: {
            "in_family_name": pair["in_family_name"],
            "held_out_name": pair["held_out_name"],
            "in_family": _drop_heavy(pair["in_family"]),
            "held_out": _drop_heavy(pair["held_out"]),
        }
        for tag, pair in runs.items()
    }
    out = Path(args.results) / dataset
    save_json({"dataset": dataset, "runs": slim}, out / "heldout.json")
    md = (
        f"# Held-out constraint: {dataset}\n\n"
        f"In-family = `{list(runs.values())[0]['in_family_name']}`. "
        f"Held-out = `{family}`. Frozen SIDs; no retraining.\n\n"
        + "\n".join(heldout_table(runs))
    )
    (out / "heldout.md").write_text(md)
    print("\n" + md)
    fig_path = Path(args.figures) / dataset / "fig7_heldout.png"
    fig_heldout(runs, fig_path, dataset=dataset)
    print(f"wrote {out / 'heldout.json'}")
    print(f"wrote {out / 'heldout.md'}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
