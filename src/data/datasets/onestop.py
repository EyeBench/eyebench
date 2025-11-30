"""Data module for creating the data."""

from __future__ import annotations

from src.data.datasets.base_dataset import ETDataset


class OneStopDataset(ETDataset):
    """Dataset for OneStop; inherits ETDataset and doesn't require a custom __init__."""
