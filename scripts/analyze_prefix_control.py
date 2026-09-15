#!/usr/bin/env python
"""Part A: prefix purity and the control realisability frontier. No GPU needed.

This is the go/no-go experiment. If a real RQ tokenizer sits close to the
``category`` upper bound, prefix masking is a usable control surface and the
paper reports a mechanism. If it sits close to the ``random`` lower bound,
hierarchical Semantic IDs do not give controllability for free, and the paper
reports that instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.analysis import analyze_tokenizer  # noqa: E402
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.sid import SIDTokenizer  # noqa: E402
from sidctl.utils import load_config, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    art = Path(args.artifacts) / dataset
    corpus = Corpus.load(art / "corpus.pkl")
    attributes = corpus.attributes
    attrs = attributes.controllable_attrs()

    print(
        f"dataset={dataset}  items={corpus.num_items}  "
        f"attrs_tested={len(attrs)}\n"
        f"labels_per_item={attributes.labels_per_item():.3f}  "
        f"multi_label_items={attributes.cooccurrence_rate():.3f}"
    )
    if attributes.labels_per_item() > 1.01:
        print(
            "  note: attribute is not a partition, so exact prefix-level\n"
            "        control is unattainable even for a perfect tokenizer"
        )
    if not attrs:
        raise SystemExit(
            "no attributes in the controllable prevalence band; loosen "
            "min/max prevalence or pick a different attribute family"
        )

    all_results = {}
    table_rows = []

    for spec in cfg["tokenizers"]:
        tag = f"{spec['kind']}_{spec['text_source']}"
        tok_path = art / tag / "tokenizer.pkl"
        if not tok_path.exists():
            print(f"[{tag}] missing tokenizer, run prepare_data.py first")
            continue

        tok = SIDTokenizer.load(tok_path)
        print(f"\n[{tag}] analysing ...")
        res = analyze_tokenizer(tok, attributes, attrs=attrs)
        res["text_source"] = spec["text_source"]
        all_results[tag] = res

        for p in res["purity_by_level"]:
            level = p["level"]
            agg = res["aggregate_by_level"][level]
            row = {
                "tokenizer": tag,
                "level": level,
                "prefixes": p["num_prefixes"],
                "size": round(p["mean_prefix_size"], 1),
                "purity": round(p["purity"], 4),
                "ami": round(p["ami"], 4),
                "spread": round(p["mean_attr_spread"], 4),
                "collateral@0leak": round(
                    agg["mean_collateral_at_zero_leakage"], 4
                ),
                "collateral@1%leak": round(
                    _mean_at(res, level, "collateral_at_leakage_01"), 4
                ),
            }
            table_rows.append(row)

    _print_table(table_rows)

    out = Path(args.results) / dataset
    save_json(
        {
            "dataset": dataset,
            "num_items": corpus.num_items,
            "attrs_tested": [attributes.names[a] for a in attrs],
            "labels_per_item": attributes.labels_per_item(),
            "multi_label_items": attributes.cooccurrence_rate(),
            "table": table_rows,
            "detail": all_results,
        },
        out / "part_a_prefix_control.json",
    )
    print(f"\nwrote {out / 'part_a_prefix_control.json'}")


def _mean_at(res: dict, level: int, key: str) -> float:
    import numpy as np

    vals = [
        s[key] for s in res["frontier_summaries"]
        if s["level"] == level and np.isfinite(s[key])
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    widths = {
        c: max(len(c), *(len(f"{r[c]}") for r in rows)) for c in cols
    }
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(f"{r[c]}".ljust(widths[c]) for c in cols))
    print(
        "\ncollateral@0leak = fraction of acceptable catalog destroyed to fully\n"
        "honour an 'avoid this attribute' request by banning SID prefixes.\n"
        "0.0 means free control; values near 1.0 mean prefix control is a fiction."
    )


if __name__ == "__main__":
    main()
