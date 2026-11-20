# SPDX-License-Identifier: Apache-2.0
"""Greedy codec sampling policy used by the Armenian evaluation runs."""

from __future__ import annotations

import torch

DEFAULT_REPETITION_PENALTY = 1.1


def penalize_repeated_codes(
    logits: torch.Tensor,
    seen_codes: torch.Tensor,
    *,
    penalty: float = DEFAULT_REPETITION_PENALTY,
) -> torch.Tensor:
    """Apply the evaluation's sign-aware penalty to previously seen IDs."""
    if logits.shape != seen_codes.shape:
        raise ValueError(
            f"logits and seen-code mask must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(seen_codes.shape)}"
        )
    if penalty <= 0:
        raise ValueError(f"repetition penalty must be positive, got {penalty}")

    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen_codes, penalized, logits)


__all__ = ["DEFAULT_REPETITION_PENALTY", "penalize_repeated_codes"]
