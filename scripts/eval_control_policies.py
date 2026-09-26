#!/usr/bin/env python
"""P0 vs Pτ vs Pmaj on frozen SID prefixes. CPU only, no retraining.

This is the locked first experiment: keep the codebook, change only the
control policy. Remapping L1 token IDs is a no-op for these metrics and is
not implemented.

    python scripts/eval_control_policies.py --config configs/amazon_beauty.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.analysis.policies import (  # noqa: E402
    diagnose_l1,
    evaluate_tokenizer_policies,
    fig_policies,
    table_rows,
)
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.sid import SIDTokenizer  # noqa: E402
from sidctl.utils import load_config, save_json  # noqa: E402


_KNOWN_TAG_PREFIXES = ("rq_", "category_", "tiled_", "random_")


def _discover_tags(art: Path, cfg: dict, requested: list[str] | None) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for spec in cfg.get("tokenizers", []):
        tag = f"{spec['kind']}_{spec['text_source']}"
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    if art.exists():
        for path in sorted(art.glob("*/tokenizer.pkl")):
            tag = path.parent.name
            if tag in seen:
                continue
            if not tag.startswith(_KNOWN_TAG_PREFIXES):
                continue
            tags.append(tag)
            seen.add(tag)
    if requested:
        allow = set(requested)
        tags = [t for t in tags if t in allow]
    return tags


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("no policy rows (missing tokenizers?)")
        return
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(f"{r[c]}") for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(f"{r[c]}".ljust(widths[c]) for c in cols))
    print(
        "\nP0 = zero-leak union (coll@0leak).  "
        "Pmaj = ban prefixes whose majority attribute is forbidden.  "
        "Pτ@0.5 = ban if P(forbidden|prefix) ≥ 0.5.\n"
        "coverage = 1 − leakage (share of forbidden items the mask actually hits)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1],
        help="prefix depths to evaluate (default: L1 only)",
    )
    ap.add_argument(
        "--tokenizers",
        nargs="+",
        default=None,
        help="restrict to these tags (default: config + any extra tokenizer.pkl)",
    )
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

    tags = _discover_tags(art, cfg, args.tokenizers)
    print(
        f"dataset={dataset}  items={corpus.num_items}  "
        f"attrs_tested={len(attrs)}  levels={args.levels}"
    )

    reports: dict[str, dict] = {}
    rows: list[dict] = []
    for tag in tags:
        tok_path = art / tag / "tokenizer.pkl"
        if not tok_path.exists():
            print(f"[{tag}] missing tokenizer, skip")
            continue
        try:
            tok = SIDTokenizer.load(tok_path)
        except (ValueError, KeyError, OSError) as exc:
            print(f"[{tag}] cannot load tokenizer ({exc}); skip")
            continue
        print(f"\n[{tag}] P0 / Pτ / Pmaj ...")
        report = evaluate_tokenizer_policies(
            tok, attributes, attrs=attrs, levels=args.levels
        )
        reports[tag] = report
        rows.extend(table_rows(tag, report))

    _print_table(rows)
    diagnosis = diagnose_l1(reports)
    print("\n" + diagnosis)

    out = Path(args.results) / dataset
    payload = {
        "dataset": dataset,
        "attrs_tested": [attributes.names[a] for a in attrs],
        "levels": args.levels,
        "table": rows,
        "diagnosis": diagnosis,
        "detail": reports,
    }
    save_json(payload, out / "control_policies.json")
    md = (
        f"# Control policies: {dataset}\n\n"
        f"Frozen SID prefixes. No retraining.\n\n"
        + _md_table(rows)
        + "\n\n"
        + diagnosis
        + "\n"
    )
    (out / "control_policies.md").write_text(md)
    print(f"\nwrote {out / 'control_policies.json'}")
    print(f"wrote {out / 'control_policies.md'}")

    if not args.no_figure and reports:
        fig_dir = Path(args.figures) / dataset
        for level in args.levels:
            path = fig_policies(reports, fig_dir, dataset, level=level)
            if path is not None:
                print(f"wrote {path}")


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "(no results)\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
