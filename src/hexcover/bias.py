"""Registry-driven spatial-bias features and scorer."""
from __future__ import annotations

from typing import Any, Callable

import h3
import numpy as np

from .config import BiasConfig, LimitsConfig
from .decorators import register

BIAS_FEATURES: dict[str, Callable[..., dict[str, float]]] = {}


@register(BIAS_FEATURES, "sri_entropy")
def _sri_entropy(centroids: np.ndarray, counts: np.ndarray,
                 cfg: BiasConfig) -> dict[str, float]:
    """Shannon-entropy Spatial Representativeness Index over grid counts."""
    flat = counts.ravel().astype(np.float64)
    total = flat.sum()
    assert total > 0, "sri_entropy requires non-empty counts"
    probs = flat[flat > 0] / total
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(counts.size))
    sri = entropy / max_entropy if max_entropy > 0 else 0.0
    return {"sri_score": sri, "entropy": entropy, "max_entropy": max_entropy}


@register(BIAS_FEATURES, "gini")
def _gini(centroids: np.ndarray, counts: np.ndarray,
          cfg: BiasConfig) -> dict[str, float]:
    """Gini coefficient of the grid-count distribution (0 = uniform)."""
    flat = np.sort(counts.ravel().astype(np.float64))
    total = flat.sum()
    assert total > 0, "gini requires non-empty counts"
    n = flat.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    gini = float((2.0 * np.sum(ranks * flat) / (n * total)) - (n + 1) / n)
    return {"gini": gini}


@register(BIAS_FEATURES, "occupancy")
def _occupancy(centroids: np.ndarray, counts: np.ndarray,
               cfg: BiasConfig) -> dict[str, float]:
    """Fraction of grid bins containing at least one centroid."""
    assert counts.size > 0, "occupancy requires a populated grid"
    return {"occupancy": float(np.count_nonzero(counts) / counts.size)}


@register(BIAS_FEATURES, "dispersion")
def _dispersion(centroids: np.ndarray, counts: np.ndarray,
                cfg: BiasConfig) -> dict[str, float]:
    """Mean centroid distance from the mean centre over the bbox diagonal."""
    assert len(centroids) > 0, "dispersion requires centroids"
    centre = centroids.mean(axis=0)
    mean_dist = float(np.linalg.norm(centroids - centre, axis=1).mean())
    span = centroids.max(axis=0) - centroids.min(axis=0)
    diag = max(float(np.linalg.norm(span)), cfg.degenerate_span)
    return {"dispersion": mean_dist / diag}


class BiasScorer:
    """Compute the configured set of spatial-bias features for a cell set."""

    def __init__(self, config: BiasConfig, limits: LimitsConfig) -> None:
        """Validate selected features against the registry and store config.

        Args:
            config (BiasConfig): Feature selection and grid settings.
            limits (LimitsConfig): Cell-count cap.

        Raises:
            KeyError: If a selected feature is not registered.
        """
        unknown = set(config.features) - set(BIAS_FEATURES)
        if unknown:
            raise KeyError(
                f"unknown bias features {sorted(unknown)};"
                f" available: {sorted(BIAS_FEATURES)}"
            )
        self._cfg = config
        self._limits = limits

    def score(self, cells: set[str]) -> dict[str, Any]:
        """Score one H3 cell set with every selected feature.

        Args:
            cells (set[str]): H3 cell indices.

        Returns:
            dict[str, Any]: Rounded metric values plus grid metadata.
        """
        assert len(cells) <= self._limits.max_bias_cells, "cell cap exceeded"
        if not cells:
            return {"n_cells": 0, "grid_counts": []}

        centroids = np.fromiter(
            (h3.cell_to_latlng(cell) for cell in cells),
            dtype=np.dtype((np.float64, 2)),
        )
        counts = self._bin(centroids)
        places = self._cfg.decimal_places
        report: dict[str, Any] = {
            "n_cells": int(len(cells)),
            "grid_counts": counts.tolist(),
        }
        for name in self._cfg.features:
            metrics = BIAS_FEATURES[name](centroids, counts, self._cfg)
            report.update(
                {key: round(val, places) for key, val in metrics.items()}
            )
        return report

    def _bin(self, centroids: np.ndarray) -> np.ndarray:
        """Bin centroids into the configured grid via ``np.histogram2d``."""
        lat, lon = centroids[:, 0], centroids[:, 1]
        span = self._cfg.degenerate_span
        lat_edges = np.linspace(
            lat.min(), max(lat.max(), lat.min() + span),
            self._cfg.grid_rows + 1,
        )
        lon_edges = np.linspace(
            lon.min(), max(lon.max(), lon.min() + span),
            self._cfg.grid_cols + 1,
        )
        counts, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
        return counts.astype(np.int64)