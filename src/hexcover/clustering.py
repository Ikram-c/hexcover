"""Single-linkage clustering and area-to-resolution mapping."""
from __future__ import annotations

import heapq

import numpy as np

from .config import ClusterConfig, HierarchyConfig, LimitsConfig
from .utils import UnionFind


class PointClusterer:
    """Single-linkage clustering via chunked vectorised pair search."""

    def __init__(self, config: ClusterConfig, limits: LimitsConfig) -> None:
        """Store clustering settings and hard limits."""
        self._cfg = config
        self._limits = limits

    def cluster(self, points: np.ndarray) -> list[np.ndarray]:
        """Return clusters meeting ``min_cluster_size``, largest-N optional.

        Args:
            points (np.ndarray): (N, 2) (lat, lon) array.

        Returns:
            list[np.ndarray]: One sub-array per cluster.
        """
        n = len(points)
        assert n <= self._limits.max_cluster_points, "cluster point cap"
        if n == 0:
            return []

        uf = UnionFind(n)
        eps_sq = self._cfg.distance_threshold_deg ** 2
        chunk = self._cfg.pair_chunk_size
        assert chunk > 0, "pair_chunk_size must be positive"

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            diff = points[start:stop, None, :] - points[None, :, :]
            dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
            rows, cols = np.nonzero(dist_sq <= eps_sq)
            for row, col in zip(rows, cols):
                i = start + int(row)
                if int(col) > i:
                    uf.union(i, int(col))

        clusters = [
            points[indices]
            for indices in uf.groups().values()
            if len(indices) >= self._cfg.min_cluster_size
        ]
        top_n = self._cfg.top_n_clusters
        if top_n is None:
            return clusters
        return heapq.nlargest(top_n, clusters, key=len)


class ResolutionMapper:
    """Map cluster hull area in square degrees to an H3 resolution."""

    def __init__(self, config: HierarchyConfig) -> None:
        """Validate threshold/resolution pairing and store the config.

        Args:
            config (HierarchyConfig): Threshold and resolution settings.

        Raises:
            ValueError: If resolutions length != thresholds length + 1.
        """
        expected = len(config.area_thresholds_sq_deg) + 1
        if len(config.resolutions) != expected:
            raise ValueError(
                "resolutions length must equal area_thresholds length + 1"
            )
        self._pairs = tuple(
            zip(config.area_thresholds_sq_deg, config.resolutions)
        )
        self._fallback = config.resolutions[-1]

    def resolution_for_area(self, area_sq_deg: float) -> int:
        """Return the H3 resolution for a hull of the given area."""
        assert area_sq_deg >= 0.0, "area must be non-negative"
        for threshold, res in self._pairs:
            if area_sq_deg < threshold:
                return res
        return self._fallback