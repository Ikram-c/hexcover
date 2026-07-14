"""GJK collision detection and bbox-grouped overlap testing."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .config import GJKConfig, PrecisionConfig
from .utils import UnionFind

Bbox = tuple[float, float, float, float]


class GJKTester:
    """2D Gilbert-Johnson-Keerthi solver with optional Nesterov acceleration."""

    def __init__(self, config: GJKConfig) -> None:
        """Store the GJK tuning parameters."""
        self._cfg = config

    def test_hulls(
        self, hull_a: np.ndarray, hull_b: np.ndarray,
    ) -> tuple[bool, float, int]:
        """Test two (lat, lon) hulls for intersection.

        Args:
            hull_a (np.ndarray): First hull.
            hull_b (np.ndarray): Second hull.

        Returns:
            tuple[bool, float, int]: (intersects, distance, iterations).
        """
        assert len(hull_a) >= 3 and len(hull_b) >= 3, "hulls need 3+ vertices"
        verts_a = np.ascontiguousarray(hull_a[:, ::-1], dtype=np.float64)
        verts_b = np.ascontiguousarray(hull_b[:, ::-1], dtype=np.float64)
        return self._gjk_2d(self._support(verts_a), self._support(verts_b))

    @staticmethod
    def _support(vertices: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Return the polygon support function for ``vertices``."""
        def support(direction: np.ndarray) -> np.ndarray:
            return vertices[int(np.argmax(vertices @ direction))].copy()
        return support

    def _gjk_2d(
        self, support_a: Callable, support_b: Callable,
    ) -> tuple[bool, float, int]:
        """Run the bounded GJK loop on two support functions."""
        ray = np.array([1.0, 0.0])
        ray_len, ray_dir = 1.0, ray.copy()
        support_point = ray.copy()
        simplex = np.zeros((3, 2), dtype=np.float64)
        simplex_len, alpha = 0, 0.0
        nesterov = self._cfg.use_nesterov

        for iteration in range(self._cfg.max_iterations):
            if ray_len < self._cfg.tolerance:
                return True, 0.0, iteration

            if nesterov:
                momentum = (iteration + 1) / (iteration + 3)
                y_vec = momentum * ray + (1.0 - momentum) * support_point
                ray_dir = momentum * ray_dir + (1.0 - momentum) * y_vec
            else:
                ray_dir = ray.copy()

            w_vec = support_a(-ray_dir) - support_b(ray_dir)
            simplex[simplex_len] = w_vec
            support_point = w_vec.copy()
            simplex_len += 1

            omega = np.dot(ray_dir, w_vec) / max(
                np.linalg.norm(ray_dir), self._cfg.norm_floor,
            )
            alpha = max(alpha, omega)
            converged = (
                iteration > 0
                and (ray_len - alpha) <= self._cfg.tolerance * ray_len
            )
            if converged:
                if nesterov:
                    nesterov, simplex_len = False, simplex_len - 1
                    continue
                return False, ray_len, iteration

            if nesterov:
                gap = 2.0 * np.dot(ray, ray - w_vec)
                if gap - self._cfg.tolerance <= 0:
                    nesterov, simplex_len = False, simplex_len - 1
                    continue

            ray, simplex_len, inside = self._advance(simplex, simplex_len)
            if inside:
                return True, 0.0, iteration
            ray_len = float(np.linalg.norm(ray))
            if ray_len == 0:
                return True, 0.0, iteration

        return False, ray_len, self._cfg.max_iterations

    @classmethod
    def _advance(
        cls, simplex: np.ndarray, simplex_len: int,
    ) -> tuple[np.ndarray, int, bool]:
        """Advance the simplex by projecting the origin onto it."""
        assert 1 <= simplex_len <= 3, "invalid simplex length"
        if simplex_len == 1:
            return simplex[0].copy(), 1, False
        if simplex_len == 2:
            return cls._project_line(simplex)
        return cls._project_triangle(simplex)

    @staticmethod
    def _project_line(simplex: np.ndarray) -> tuple[np.ndarray, int, bool]:
        """Project the origin onto a 2-vertex simplex."""
        a_pt, b_pt = simplex[1], simplex[0]
        ab = b_pt - a_pt
        dot_toward = np.dot(ab, -a_pt)
        if dot_toward <= 0:
            simplex[0] = a_pt.copy()
            return a_pt.copy(), 1, bool(np.allclose(a_pt, 0))
        simplex[0], simplex[1] = b_pt.copy(), a_pt.copy()
        dot_end = np.dot(ab, b_pt)
        ray = (dot_end * a_pt + dot_toward * b_pt) / np.dot(ab, ab)
        return ray, 2, False

    @staticmethod
    def _triple_product(
        u: np.ndarray, v: np.ndarray, w: np.ndarray,
    ) -> np.ndarray:
        """Return ``v * (u . w) - u * (v . w)``."""
        return v * np.dot(u, w) - u * np.dot(v, w)

    @classmethod
    def _project_triangle(
        cls, simplex: np.ndarray,
    ) -> tuple[np.ndarray, int, bool]:
        """Project the origin onto a 3-vertex simplex."""
        a_pt, b_pt, c_pt = simplex[2], simplex[1], simplex[0]
        ab, ac, ao = b_pt - a_pt, c_pt - a_pt, -a_pt

        ab_perp = cls._triple_product(ac, ab, ab)
        if np.dot(ab_perp, ao) > 0:
            return cls._edge_or_vertex(simplex, a_pt, b_pt, ab)
        ac_perp = cls._triple_product(ab, ac, ac)
        if np.dot(ac_perp, ao) > 0:
            return cls._edge_or_vertex(simplex, a_pt, c_pt, ac)

        simplex[0], simplex[1], simplex[2] = (
            c_pt.copy(), b_pt.copy(), a_pt.copy(),
        )
        return np.zeros(2), 3, True

    @staticmethod
    def _edge_or_vertex(
        simplex: np.ndarray,
        a_pt: np.ndarray,
        end_pt: np.ndarray,
        edge: np.ndarray,
    ) -> tuple[np.ndarray, int, bool]:
        """Project the origin onto an edge or one of its endpoints."""
        dot_toward = np.dot(edge, -a_pt)
        if dot_toward <= 0:
            simplex[0] = a_pt.copy()
            return a_pt.copy(), 1, False
        simplex[0], simplex[1] = end_pt.copy(), a_pt.copy()
        dot_end = np.dot(edge, end_pt)
        ray = (dot_end * a_pt + dot_toward * end_pt) / np.dot(edge, edge)
        return ray, 2, False


class OverlapDetector:
    """Bbox component grouping plus per-pair GJK testing for named hulls."""

    def __init__(self, tester: GJKTester, precision: PrecisionConfig) -> None:
        """Store the GJK tester and reporting precision."""
        self._gjk = tester
        self._precision = precision

    @staticmethod
    def bbox_of_hull(hull: np.ndarray) -> Bbox:
        """Return (min_lon, min_lat, max_lon, max_lat) for a (lat, lon) hull."""
        assert len(hull) > 0, "empty hull"
        return (
            float(hull[:, 1].min()), float(hull[:, 0].min()),
            float(hull[:, 1].max()), float(hull[:, 0].max()),
        )

    @staticmethod
    def _bboxes_overlap(a: Bbox, b: Bbox) -> bool:
        """Return True if two bboxes overlap on both axes."""
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    def find_groups(self, bboxes: list[Bbox]) -> list[tuple[int, ...]]:
        """Cluster bbox indices into connected overlap components.

        Args:
            bboxes (list[Bbox]): One bbox per hull.

        Returns:
            list[tuple[int, ...]]: Components with 2+ members.
        """
        count = len(bboxes)
        uf = UnionFind(count)
        for i in range(count):
            for j in range(i + 1, count):
                if self._bboxes_overlap(bboxes[i], bboxes[j]):
                    uf.union(i, j)
        return [
            tuple(members)
            for members in uf.groups().values()
            if len(members) >= 2
        ]

    def test_pairs(
        self,
        groups: list[tuple[int, ...]],
        names: list[str],
        hulls: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        """Run one GJK test per pair within each component."""
        results: list[dict[str, Any]] = []
        for group in groups:
            for pos_a in range(len(group)):
                for pos_b in range(pos_a + 1, len(group)):
                    results.append(self._test_pair(
                        names[group[pos_a]], names[group[pos_b]], hulls,
                    ))
        return results

    def _test_pair(
        self, name_a: str, name_b: str, hulls: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """GJK-test one named pair and return a result record."""
        hit, dist, iters = self._gjk.test_hulls(hulls[name_a], hulls[name_b])
        status = "INTERSECT" if hit else f"dist={dist:.6f}"
        print(f"  GJK {name_a} / {name_b}: {status} ({iters} iters)")
        return {
            "a": name_a,
            "b": name_b,
            "intersects": hit,
            "distance": round(dist, self._precision.distance_decimal_places),
        }