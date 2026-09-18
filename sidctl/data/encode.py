"""Shared history encoding, so training and evaluation cannot drift apart."""

from __future__ import annotations

import torch

from sidctl.data.vocab import Vocab
from sidctl.sid.tokenizer import SIDTokenizer


def encode_history(
    events: list[dict],
    tokenizer: SIDTokenizer,
    vocab: Vocab,
    max_history_len: int = 20,
    include_polarity: bool = True,
    positives_only: bool = True,
) -> list[int]:
    """
    Flatten a user's event list into SID token ids.

    Each event contributes an optional polarity marker followed by the item's
    SID tokens. Only the most recent ``max_history_len`` events are kept.
    """
    if positives_only:
        events = [e for e in events if e["polarity"] == "pos"]
    events = events[-max_history_len:]

    tokens: list[int] = []
    for ev in events:
        if include_polarity:
            tokens.append(
                vocab.pos_id if ev["polarity"] == "pos" else vocab.neg_id
            )
        tokens.extend(vocab.sid_to_ids(tokenizer.get_sid(ev["item_idx"])))
    return tokens


def to_tensors(
    tokens: list[int], device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return input_ids, torch.ones_like(input_ids)
