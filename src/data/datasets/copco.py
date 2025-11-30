"""Data module for creating the data."""

from __future__ import annotations

from src.data.datasets.base_dataset import ETDataset


class CopCoDataset(ETDataset):
    """Dataset for CopCo; inherits ETDataset and doesn't require a custom __init__."""
