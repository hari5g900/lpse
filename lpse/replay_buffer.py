"""Shared replay buffer used by the LPSE, MLP, and XGBoost trainers.

Context tensors are interned in a separate FIFO map keyed by id, so a single
snapshot is stored only once even when many QA pairs reference it. The live
entry count is tracked incrementally so `len()` is O(1) instead of rescanning
the (potentially 32k-entry) QA deque on every step.
"""

from __future__ import annotations

import random
from collections import OrderedDict, deque

import torch


class QAReplayBuffer:
    """Stores (context_id, question, answer_idx) tuples with interned contexts."""

    def __init__(self, qa_capacity: int, context_capacity: int):
        self.qa_capacity = qa_capacity
        self.context_capacity = context_capacity
        self.contexts: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.qa_entries: deque[tuple[int, dict, int]] = deque(maxlen=qa_capacity)
        self._next_id = 0

        self._live = 0
        self._ctx_counts: dict[int, int] = {}

    def add_context(self, context_seq: torch.Tensor) -> int:
        cid = self._next_id
        self._next_id += 1
        self.contexts[cid] = context_seq.detach().cpu()
        while len(self.contexts) > self.context_capacity:
            evicted, _ = self.contexts.popitem(last=False)
            self._live -= self._ctx_counts.get(evicted, 0)
            self._ctx_counts[evicted] = 0
        return cid

    def add_qa(self, context_id: int, question: dict, answer_idx: int):
        if self.qa_entries and len(self.qa_entries) >= self.qa_capacity:
            dropped_cid = self.qa_entries[0][0]
            if dropped_cid in self.contexts:
                self._live -= 1

            self._ctx_counts[dropped_cid] = self._ctx_counts.get(dropped_cid, 0) - 1
        self.qa_entries.append((context_id, question, answer_idx))
        self._ctx_counts[context_id] = self._ctx_counts.get(context_id, 0) + 1
        self._live += 1

    def __len__(self) -> int:
        return self._live

    def sample(self, batch_size: int):
        candidates = [e for e in self.qa_entries if e[0] in self.contexts]
        if not candidates:
            raise ValueError("Replay buffer has no live entries to sample.")
        bs = min(batch_size, len(candidates))
        chosen = random.sample(candidates, bs)
        seqs, qs, ans = [], [], []
        for cid, q, a in chosen:
            seqs.append(self.contexts[cid])
            qs.append(q)
            ans.append(a)
        return (
            torch.stack(seqs, dim=0),
            qs,
            torch.tensor(ans, dtype=torch.long),
        )

    def all_live(self) -> list[tuple[torch.Tensor, dict, int]]:
        return [
            (self.contexts[cid], q, a)
            for cid, q, a in self.qa_entries
            if cid in self.contexts
        ]
