#!/usr/bin/env python
"""Part B: cost of enforcing an attribute constraint at decoding time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.control import (  # noqa: E402
    DEFAULT_DECODERS,
    POLICY_DECODER_NAMES,
    PRUNE_DECODER_NAMES,
    SEARCH_DECODER_NAMES,
    build_control_instances,
    instance_stats,
    select_decoders,
)
from sidctl.data import build_vocab  # noqa: E402
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.eval import evaluate_control  # noqa: E402
from sidctl.sid import SIDTokenizer  # noqa: E402
from sidctl.train import Trainer  # noqa: E402
from sidctl.utils import load_config, pick_device, save_json, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--checkpoint-name", default="best.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument(
        "--decoders",
        default=None,
        help="comma-separated decoder names. "
        f"'policy' expands to {','.join(POLICY_DECODER_NAMES)}; "
        "'search' expands to the oracle / beam-sweep / PACD / lookahead study; "
        "'prune' expands to the constraint-aware pruning theta sweep",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="output suffix: control_eval_<tag>.json (default: search preset -> 'search')",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = pick_device(args.device or cfg.get("device"))

    dataset = cfg["dataset"]
    art = Path(args.artifacts) / dataset
    corpus = Corpus.load(art / "corpus.pkl")
    tok = SIDTokenizer.load(art / args.tokenizer / "tokenizer.pkl")
    vocab = build_vocab(tok)

    ckpt_path = Path(args.checkpoints) / dataset / args.tokenizer / args.checkpoint_name
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint {ckpt_path}; run train.py first")
    model, _ = Trainer.load_checkpoint(ckpt_path, device="cpu")

    ccfg = cfg.get("control", {})
    instances = build_control_instances(
        corpus,
        allowed_attrs=corpus.attributes.controllable_attrs(),
        min_attr_count=ccfg.get("min_attr_count", 3),
        min_history=ccfg.get("min_history", 3),
        max_users=args.max_users or ccfg.get("max_users"),
        split="test",
        seed=cfg.get("seed", 42),
    )
    stats = instance_stats(instances)
    print("control instances:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if not instances:
        raise SystemExit("no eligible control instances")

    if args.decoders is None:
        specs = list(DEFAULT_DECODERS)
    elif args.decoders.strip() == "policy":
        specs = select_decoders(list(POLICY_DECODER_NAMES))
    elif args.decoders.strip() == "search":
        specs = select_decoders(list(SEARCH_DECODER_NAMES))
        args.tag = args.tag or "search"
    elif args.decoders.strip() == "prune":
        specs = select_decoders(list(PRUNE_DECODER_NAMES))
        args.tag = args.tag or "prune"
    else:
        specs = select_decoders(
            [n.strip() for n in args.decoders.split(",") if n.strip()]
        )
    print("decoders: " + ", ".join(s.name for s in specs))

    results = evaluate_control(
        model=model,
        tokenizer=tok,
        vocab=vocab,
        corpus=corpus,
        instances=instances,
        specs=specs,
        beam_size=args.beam_size or ccfg.get("beam_size", 50),
        topk=ccfg.get("topk", 10),
        max_history_len=cfg.get("train", {}).get("max_history_len", 20),
        device=device,
    )
    results["instance_stats"] = stats
    results["tokenizer"] = args.tokenizer

    _print(results)
    out = Path(args.results) / dataset / args.tokenizer
    rows = results.pop("rows")
    stem = f"control_eval_{args.tag}" if args.tag else "control_eval"
    save_json(results, out / f"{stem}.json")
    save_json(rows, out / f"{stem}_rows.json")
    print(f"\nwrote {out / (stem + '.json')}")


def _print(results: dict) -> None:
    k = results["topk"]
    print(
        f"\n{'decoder':<24s} {'NDCG@'+str(k):>9s} {'retain':>7s} "
        f"{'viol@'+str(k):>8s} {'anyviol':>8s} {'short':>7s} {'fill':>6s} "
        f"{'overlap':>8s} {'missed':>7s} {'ms':>8s}"
    )
    print("-" * 104)
    for name, m in results["per_decoder"].items():
        print(
            f"{name:<24s} {m['ndcg']:>9.4f} {m['ndcg_retention']:>7.3f} "
            f"{m['violation_rate']:>8.4f} {m['any_violation']:>8.3f} "
            f"{m['short_list_rate']:>7.3f} {m['fill_rate']:>6.3f} "
            f"{m.get('oracle_overlap', float('nan')):>8.3f} "
            f"{m.get('missed_best', float('nan')):>7.3f} "
            f"{m['latency_ms']:>8.1f}"
        )
    print(
        "\nEvery constraint is compatible with the held-out target, so an ideal\n"
        "control surface would show retain=1.000, viol=0.000, short=0.000."
    )


if __name__ == "__main__":
    main()
