"""Residual K-Means quantization (TIGER-style hierarchical SIDs)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class RQKMeans:
    """Hierarchical residual quantization with KMeans at each level."""

    def __init__(
        self,
        num_levels: int = 3,
        codebook_size: int = 256,
        random_state: int = 42,
    ):
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.codebooks: list[np.ndarray] = []

    def fit(self, embeddings: np.ndarray) -> "RQKMeans":
        residual = embeddings.astype(np.float64).copy()
        self.codebooks = []
        for _ in range(self.num_levels):
            km = KMeans(
                n_clusters=self.codebook_size,
                random_state=self.random_state,
                n_init=3,
                max_iter=100,
            )
            codes = km.fit_predict(residual)
            centers = km.cluster_centers_
            self.codebooks.append(centers)
            residual = residual - centers[codes]
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Return (N, num_levels) integer codes."""
        residual = embeddings.astype(np.float64).copy()
        all_codes = []
        for level in range(self.num_levels):
            centers = self.codebooks[level]
            # Nearest center via brute force (fine for 256 x dim)
            dists = (
                (residual[:, None, :] - centers[None, :, :]) ** 2
            ).sum(axis=2)
            codes = dists.argmin(axis=1)
            all_codes.append(codes)
            residual = residual - centers[codes]
        return np.stack(all_codes, axis=1).astype(np.int64)

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct embeddings from code indices."""
        recon = np.zeros((codes.shape[0], self.codebooks[0].shape[1]))
        for level in range(self.num_levels):
            recon += self.codebooks[level][codes[:, level]]
        return recon
