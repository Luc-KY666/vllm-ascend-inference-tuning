"""Qwen control-output helpers for timing-only simulation."""

from __future__ import annotations


def materialize_sampled_ids(
    batch_size: int,
    width: int = 8,
    token_id: int = 1,
) -> list[list[int]]:
    """Return padded sampled-ID rows with only column zero populated."""
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if width < 1:
        raise ValueError("width must be positive")
    if token_id < 0:
        raise ValueError("token_id must be non-negative")
    return [[token_id, *([0] * (width - 1))] for _ in range(batch_size)]
