#!/usr/bin/env python
"""Train the TIGER-style recommender for one tokenizer variant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader  # noqa: E402

from sidctl.data import GRDataset, build_vocab, collate_fn  # noqa: E402
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.models import TigerGR  # noqa: E402
from sidctl.sid import SIDTokenizer  # noqa: E402
from sidctl.train import Trainer  # noqa: E402
from sidctl.utils import load_config, pick_device, save_json, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True, help="e.g. rq_title")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = pick_device(args.device or cfg.get("device"))

    dataset = cfg["dataset"]
    art = Path(args.artifacts) / dataset
    corpus = Corpus.load(art / "corpus.pkl")
    tok = SIDTokenizer.load(art / args.tokenizer / "tokenizer.pkl")
    vocab = build_vocab(tok)

    tcfg = cfg.get("train", {})
    mcfg = cfg.get("model", {})
    max_hist = tcfg.get("max_history_len", 20)

    common = dict(
        corpus=corpus,
        tokenizer=tok,
        vocab=vocab,
        max_history_len=max_hist,
        include_negatives=tcfg.get("include_negatives", False),
    )
    train_ds = GRDataset(
        **common,
        split="train",
        max_examples_per_user=tcfg.get("max_examples_per_user"),
        tile_targets=tcfg.get("tile_targets", "all"),
    )
    val_ds = GRDataset(**common, split="val")

    print(f"device={device}  vocab={vocab.vocab_size}  sid_len={tok.sid_length}")
    print(f"train examples={len(train_ds)}  val examples={len(val_ds)}")
    if len(train_ds) == 0:
        raise SystemExit("no training examples; check the corpus filters")

    batch = tcfg.get("batch_size", 128)
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, collate_fn=collate_fn,
        num_workers=tcfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn,
        num_workers=tcfg.get("num_workers", 0),
    )

    model = TigerGR(
        vocab_size=vocab.vocab_size,
        sid_length=tok.sid_length,
        level_sizes=tok.level_sizes,
        pos_token_id=vocab.pos_id,
        neg_token_id=vocab.neg_id,
        d_model=mcfg.get("d_model", 256),
        num_layers=mcfg.get("num_layers", 4),
        num_heads=mcfg.get("num_heads", 6),
        d_ff=mcfg.get("d_ff", 1024),
        dropout=mcfg.get("dropout", 0.1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters={n_params/1e6:.2f}M")

    ckpt_dir = Path(args.checkpoints) / dataset / args.tokenizer
    trainer = Trainer(
        model=model,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=tcfg.get("lr", 3e-4),
        weight_decay=tcfg.get("weight_decay", 0.01),
        grad_clip=tcfg.get("grad_clip", 1.0),
        warmup_steps=tcfg.get("warmup_steps", 500),
        checkpoint_dir=ckpt_dir,
        device=device,
    )
    history = trainer.train(
        epochs=args.epochs or tcfg.get("epochs", 20),
        early_stop_patience=tcfg.get("early_stop_patience", 3),
    )
    save_json(
        {
            "tokenizer": args.tokenizer,
            "params": n_params,
            "train_examples": len(train_ds),
            "history": history,
            "best_val_loss": trainer.best_val_loss,
        },
        ckpt_dir / "train_summary.json",
    )
    print(f"\ncheckpoints -> {ckpt_dir}")


if __name__ == "__main__":
    main()
