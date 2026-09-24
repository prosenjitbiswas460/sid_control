"""Next-item prediction examples under the standard leave-one-out protocol.

Per user, the last positive interaction is the test target, the second to last
is the validation target, and everything earlier is available for training. This
matches the SASRec/TIGER convention, so accuracy numbers are comparable to
published SID recommenders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

from sidctl.data.corpus import Corpus
from sidctl.data.encode import encode_history
from sidctl.data.vocab import Vocab
from sidctl.sid.tokenizer import SIDTokenizer

Split = Literal["train", "val", "test"]


@dataclass
class GRExample:
    input_ids: list[int]
    labels: list[int]


def split_positions(events: list[dict]) -> tuple[list[int], int | None, int | None]:
    """Positive event positions, plus the validation and test cut points."""
    pos = [i for i, e in enumerate(events) if e["polarity"] == "pos"]
    if len(pos) < 3:
        return pos, None, None
    return pos, pos[-2], pos[-1]


class GRDataset(Dataset):
    def __init__(
        self,
        corpus: Corpus,
        tokenizer: SIDTokenizer,
        vocab: Vocab,
        split: Split = "train",
        max_history_len: int = 20,
        min_history: int = 1,
        include_negatives: bool = False,
        max_examples_per_user: int | None = None,
        user_ids: list[int] | None = None,
        tile_targets: Literal["primary", "all"] = "primary",
    ):
        self.examples: list[GRExample] = []
        users = user_ids if user_ids is not None else sorted(corpus.user_events)

        for uid in users:
            events = corpus.user_events[uid]
            pos, val_cut, test_cut = split_positions(events)
            if val_cut is None or test_cut is None:
                continue

            if split == "train":
                targets = [p for p in pos[:-2]]
            elif split == "val":
                targets = [val_cut]
            else:
                targets = [test_cut]

            if split == "train" and max_examples_per_user is not None:
                targets = targets[-max_examples_per_user:]

            for t in targets:
                history = events[:t]
                if sum(1 for e in history if e["polarity"] == "pos") < min_history:
                    continue
                input_ids = encode_history(
                    history,
                    tokenizer,
                    vocab,
                    max_history_len=max_history_len,
                    positives_only=not include_negatives,
                )
                if not input_ids:
                    continue
                item_idx = events[t]["item_idx"]
                if tile_targets == "all":
                    target_sids = tokenizer.iter_sids(item_idx)
                else:
                    target_sids = [tokenizer.get_sid(item_idx)]
                for sid in target_sids:
                    self.examples.append(
                        GRExample(
                            input_ids=input_ids,
                            labels=vocab.sid_to_ids(sid),
                        )
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {"input_ids": ex.input_ids, "labels": ex.labels}


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    max_in = max(len(b["input_ids"]) for b in batch)
    max_lab = max(len(b["labels"]) for b in batch)

    input_ids, labels, attn = [], [], []
    for b in batch:
        inp, lab = b["input_ids"], b["labels"]
        input_ids.append(inp + [pad_id] * (max_in - len(inp)))
        labels.append(lab + [-100] * (max_lab - len(lab)))
        attn.append([1] * len(inp) + [0] * (max_in - len(inp)))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }
