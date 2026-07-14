"""Convex hulls, shoelace areas, and H3 conversion."""
from __future__ import annotations

from typing import Iterator, Optional

import h3
import numpy as np

from .config import HullConfig, LimitsConfig


def convex_hull(points: np.ndarray, config: HullConfig,
                limits: LimitsConfig) -> np.ndarray:
    """Compute a convex hull via Andrew's monotone chain (iterative).

    Args:
        points (np.ndarray): (N, 2) (lat, lon) array.
        config (HullConfig): Hull settings.
        limits (LimitsConfig): Loop-bound limits.

    Returns:
        np.ndarray: (M, 2) hull vertices in counter-clockwise order.
    """
    assert points.ndim == 2 and points.shape[1] == 2, "expected (N, 2) input"
    assert len(points) <= limits.max_input_points, "input exceeds point cap"
    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) < 3:
        return pts

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    tol = config.collinear_tolerance
    max_ops = limits.max_hull_iterations_factor * len(pts)

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float(
            (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        )

    def _half(sequence: np.ndarray) -> list[np.ndarray]:
        chain: list[np.ndarray] = []
        ops = 0
        for point in sequence:
            while (
                len(chain) >= 2
                and _cross(chain[-2], chain[-1], point) <= tol
            ):
                chain.pop()
                ops += 1
                assert ops <= max_ops, "hull iteration bound exceeded"
            chain.append(point)
        return chain

    lower = _half(pts)
    upper = _half(pts[::-1])
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    assert len(hull) >= 3, "degenerate hull from non-degenerate input"
    return hull


def hull_area_sq_deg(hull: np.ndarray) -> float:
    """Return the absolute shoelace area of a (lat, lon) hull.

    Args:
        hull (np.ndarray): (M, 2) hull vertices.

    Returns:
        float: Area in square degrees.
    """
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 1], hull[:, 0]
    return float(
        0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


class HexConverter:
    """Convert hulls to H3 cell sets and cells to GeoJSON features."""

    def __init__(self, resolution: int, min_vertices: int) -> None:
        """Store the base resolution and minimum hull vertex count."""
        assert 0 <= resolution <= 15, "H3 resolution out of range"
        self._resolution = resolution
        self._min_vertices = min_vertices

    def hull_to_cells(
        self, hull: np.ndarray, resolution: Optional[int] = None,
    ) -> set[str]:
        """Cover a hull polygon with H3 cells.

        Args:
            hull (np.ndarray): (M, 2) (lat, lon) hull vertices.
            resolution (Optional[int]): Override resolution.

        Returns:
            set[str]: H3 cell indices.
        """
        assert len(hull) >= self._min_vertices, "hull below vertex minimum"
        res = self._resolution if resolution is None else resolution
        poly = h3.LatLngPoly([(float(p[0]), float(p[1])) for p in hull])
        return set(h3.polygon_to_cells(poly, res))

    @staticmethod
    def iter_features(cells: set[str], props: dict) -> Iterator[dict]:
        """Yield one GeoJSON polygon feature per H3 cell."""
        for cell in sorted(cells):
            boundary = h3.cell_to_boundary(cell)
            ring = [[lng, lat] for lat, lng in boundary]
            ring.append(ring[0])
            yield {
                "type": "Feature",
                "properties": {**props, "h3_index": cell},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }

    @staticmethod
    def hull_to_polygon_feature(hull: np.ndarray, props: dict) -> dict:
        """Return one GeoJSON polygon feature for a hull."""
        ring = [[float(p[1]), float(p[0])] for p in hull]
        ring.append(ring[0])
        return {
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }