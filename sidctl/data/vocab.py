"""Flat token vocabulary over per-level SID codes plus polarity markers."""

from __future__ import annotations

from dataclasses import dataclass

from sidctl.sid.tokenizer import SIDTokenizer

PAD_TOKEN = "<PAD>"
POS_TOKEN = "<POS>"
NEG_TOKEN = "<NEG>"


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    level_sizes: list[int]
    sid_length: int
    pad_id: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pos_id(self) -> int:
        return self.token_to_id[POS_TOKEN]

    @property
    def neg_id(self) -> int:
        return self.token_to_id[NEG_TOKEN]

    @property
    def level_offsets(self) -> list[int]:
        offsets, running = [], 1
        for size in self.level_sizes:
            offsets.append(running)
            running += size
        return offsets

    def sid_to_ids(self, sid: tuple[int, ...]) -> list[int]:
        offsets = self.level_offsets
        return [offsets[l] + code for l, code in enumerate(sid)]

    def ids_to_sid(self, ids: list[int]) -> tuple[int, ...] | None:
        if len(ids) != self.sid_length:
            return None
        offsets = self.level_offsets
        sid = []
        for level, tid in enumerate(ids):
            code = tid - offsets[level]
            if code < 0 or code >= self.level_sizes[level]:
                return None
            sid.append(code)
        return tuple(sid)


def build_vocab(tokenizer: SIDTokenizer) -> Vocab:
    token_to_id: dict[str, int] = {PAD_TOKEN: 0}
    idx = 1
    for level, size in enumerate(tokenizer.level_sizes):
        for code in range(size):
            token_to_id[f"L{level}_C{code}"] = idx
            idx += 1
    token_to_id[POS_TOKEN] = idx
    token_to_id[NEG_TOKEN] = idx + 1
    return Vocab(
        token_to_id=token_to_id,
        id_to_token={v: k for k, v in token_to_id.items()},
        level_sizes=tokenizer.level_sizes,
        sid_length=tokenizer.sid_length,
    )
