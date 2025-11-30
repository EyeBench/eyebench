"""This module contains utility functions for the models."""

from __future__ import annotations

import numpy as np

# Here I list functions used for the dimensionality reduction of the RoBERTa embeddings
# https://deep-ch.medium.com/dimension-reduction-by-whitening-bert-roberta-5e103093f782


def transform_and_normalize(
    vecs: np.ndarray, kernel: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Applying transformation then standardize
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)


def normalize(vecs: np.ndarray) -> np.ndarray:
    """
    Standardization
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True) ** 0.5


def compute_kernel_bias(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    vecs shape is (batch_size, max_seq_len, hidden_size)
    Calculate Kernel & Bias for the final transformation - y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(dim=0, keepdim=True)
    cov = np.cov(vecs.T)
    u, s, unused_vh = np.linalg.svd(cov)
    w = np.dot(u, np.diag(s**0.5))
    w = np.linalg.inv(w.T)
    return w, -mu
