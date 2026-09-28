#!/usr/bin/env python
"""Search-error study: allowed trie vs PACD vs lookahead against the exact oracle.

Reads results/<dataset>/<tokenizer>/control_eval_search.json for every
tokenizer given (or every one that has the file) and writes
search.md and fig5_search.png next to them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sidctl.utils import load_config  # noqa: E402

FAMILIES = ("allowed_trie", "pacd", "lookahead")


def _curves(per: dict) -> dict[str, list[tuple[int, dict]]]:
    out: dict[str, list[tuple[int, dict]]] = {f: [] for f in FAMILIES}
    for name, m in per.items():
        hit = re.fullmatch(r"(allowed_trie|pacd|lookahead)_b(\d+)", name)
        if hit:
            out[hit.group(1)].append((int(hit.group(2)), m))
    return {f: sorted(v, key=lambda x: x[0]) for f, v in out.items() if v}


def _table(tok: str, per: dict) -> list[str]:
    lines = [
        f"### {tok}",
        "",
        "| decoder | NDCG@10 | viol@10 | fill | overlap w/ oracle | missed best | ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in per.items():
        lines.append(
            f"| {name} | {m['ndcg']:.4f} | {m['violation_rate']:.3f} | "
            f"{m['fill_rate']:.3f} | {m.get('oracle_overlap', float('nan')):.3f} | "
            f"{m.get('missed_best', float('nan')):.3f} | {m['latency_ms']:.1f} |"
        )
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tokenizers", default=None, help="comma list; default: all found")
    args = ap.parse_args()

    dataset = load_config(args.config)["dataset"]
    root = Path(args.results) / dataset
    if args.tokenizers:
        toks = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
    else:
        toks = sorted(p.parent.name for p in root.glob("*/control_eval_search.json"))
    runs = {}
    for tok in toks:
        path = root / tok / "control_eval_search.json"
        if path.exists():
            runs[tok] = json.loads(path.read_text())["per_decoder"]
        else:
            print(f"skip {tok}: no {path}")
    if not runs:
        raise SystemExit(f"no control_eval_search.json under {root}")

    md = [f"# Search-error study: {dataset}", ""]
    for tok, per in runs.items():
        md += _table(tok, per)
    (root / "search.md").write_text("\n".join(md))

    fig, axes = plt.subplots(len(runs), 2, figsize=(10, 3.4 * len(runs)), squeeze=False)
    for row, (tok, per) in enumerate(runs.items()):
        ax_n, ax_o = axes[row]
        for fam, pts in _curves(per).items():
            beams = [b for b, _ in pts]
            ax_n.plot(beams, [m["ndcg"] for _, m in pts], marker="o", label=fam)
            ax_o.plot(beams, [m.get("oracle_overlap", float("nan")) for _, m in pts],
                      marker="o", label=fam)
        if "oracle_allowed" in per:
            ax_n.axhline(per["oracle_allowed"]["ndcg"], ls="--", c="k", label="oracle (allowed)")
            ax_o.axhline(1.0, ls="--", c="k")
        if "unconstrained" in per:
            ax_n.axhline(per["unconstrained"]["ndcg"], ls=":", c="grey", label="unconstrained b50")
        for ax, ylabel in ((ax_n, "NDCG@10"), (ax_o, "overlap with oracle top-10")):
            ax.set_xscale("log")
            ax.set_xlabel("beam width")
            ax.set_ylabel(ylabel)
            ax.set_title(tok)
        ax_n.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "fig5_search.png", dpi=150)
    print(f"wrote {root / 'search.md'} and {root / 'fig5_search.png'}")


if __name__ == "__main__":
    main()
