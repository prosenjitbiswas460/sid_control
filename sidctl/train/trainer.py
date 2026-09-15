"""Training loop for the SID generative recommender."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sidctl.data.vocab import Vocab
from sidctl.models.tiger import TigerGR


class Trainer:
    def __init__(
        self,
        model: TigerGR,
        vocab: Vocab,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_steps: int = 500,
        checkpoint_dir: str | Path = "checkpoints",
        device: str | None = None,
    ):
        self.model = model
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.base_lr = lr
        self.step = 0
        self.best_val_loss = float("inf")

    def _set_lr(self) -> None:
        if self.warmup_steps <= 0:
            return
        scale = min(1.0, self.step / self.warmup_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * scale

    def _loss(self, batch: dict) -> torch.Tensor:
        out = self.model(
            batch["input_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
            batch["labels"].to(self.device),
        )
        return out["loss"]

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            total += self._loss(batch).item()
            n += 1
        self.model.train()
        return total / max(n, 1)

    def train(self, epochs: int, early_stop_patience: int = 3) -> dict:
        history = {"train_loss": [], "val_loss": []}
        patience = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total, n = 0.0, 0
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}/{epochs}")
            for batch in pbar:
                self.step += 1
                self._set_lr()
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()
                total += loss.item()
                n += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total / max(n, 1)
            val_loss = self.validate()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience = 0
                self.save_checkpoint("best.pt")
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print("early stopping")
                    break

        self.save_checkpoint("last.pt")
        (self.checkpoint_dir / "history.json").write_text(
            json.dumps(history, indent=2)
        )
        return history

    def save_checkpoint(self, name: str) -> None:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "vocab_size": self.vocab.vocab_size,
                "level_sizes": self.vocab.level_sizes,
                "sid_length": self.vocab.sid_length,
                "model_config": {
                    "d_model": self.model.model.config.d_model,
                    "num_layers": self.model.model.config.num_layers,
                    "num_heads": self.model.model.config.num_heads,
                    "d_ff": self.model.model.config.d_ff,
                },
            },
            path,
        )

    @staticmethod
    def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[TigerGR, dict]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        model = TigerGR(
            vocab_size=ckpt["vocab_size"],
            sid_length=ckpt["sid_length"],
            level_sizes=ckpt["level_sizes"],
            d_model=cfg.get("d_model", 256),
            num_layers=cfg.get("num_layers", 4),
            num_heads=cfg.get("num_heads", 6),
            d_ff=cfg.get("d_ff", 1024),
        )
        model.load_state_dict(ckpt["model_state"])
        return model, ckpt
