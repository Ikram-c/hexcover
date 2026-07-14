"""Rasterise H3 pipeline outputs into a WEBKNOSSOS dataset."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import h3
import numpy as np

from .config import LimitsConfig, WebknossosConfig
from .decorators import register

WK_LAYER_BUILDERS: dict[str, Callable[..., "LayerSpec"]] = {}

COLOR_CATEGORY = "color"
SEGMENTATION_CATEGORY = "segmentation"


@dataclass(frozen=True)
class LayerSpec:
    """One WEBKNOSSOS layer ready to be written."""

    name: str
    data: np.ndarray
    category: str
    largest_segment_id: int = 0


@dataclass(frozen=True)
class ExportContext:
    """Everything the layer builders need from a finished pipeline run."""

    region_cells: dict[str, set[str]]
    region_types: dict[str, str]
    overlap_cells: dict[str, set[str]]
    cluster_cells: dict[int, tuple[int, set[str]]]
    bias_report: dict[str, Any]
    base_resolution: int
    eo_footprints: tuple = ()


class HexRasteriser:
    """Map H3 cell sets onto a fixed lat/lon pixel grid.

    The pixel-to-cell index is computed once per H3 resolution and
    cached, so rasterising many layers reuses the same lookups.
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        voxel_deg: float,
        max_pixels: int,
    ) -> None:
        """Derive the grid shape from a (S, W, N, E) bbox.

        Args:
            bbox (tuple[float, float, float, float]): (S, W, N, E) bounds.
            voxel_deg (float): Degrees per pixel on both axes.
            max_pixels (int): Hard cap on grid size (Power of 10 rule 2).

        Raises:
            ValueError: If the grid would exceed ``max_pixels``.
        """
        south, west, north, east = bbox
        assert north > south and east > west, "degenerate bbox"
        assert voxel_deg > 0, "voxel_deg must be positive"
        self._bbox = bbox
        self._voxel_deg = voxel_deg
        self.height = max(int(np.ceil((north - south) / voxel_deg)), 1)
        self.width = max(int(np.ceil((east - west) / voxel_deg)), 1)
        if self.height * self.width > max_pixels:
            raise ValueError(
                f"raster {self.width}x{self.height} exceeds"
                f" max_raster_pixels={max_pixels};"
                " increase voxel_deg or the limit"
            )
        self._cell_index: dict[int, np.ndarray] = {}

    @property
    def shape(self) -> tuple[int, int]:
        """Return the (height, width) grid shape."""
        return (self.height, self.width)

    def _pixel_cells(self, resolution: int) -> np.ndarray:
        """Return (H, W) object array of H3 indices for pixel centres."""
        assert 0 <= resolution <= 15, "H3 resolution out of range"
        cached = self._cell_index.get(resolution)
        if cached is not None:
            return cached

        south, west, _, _ = self._bbox
        step = self._voxel_deg
        cells = np.empty((self.height, self.width), dtype=object)
        for row in range(self.height):
            lat = south + (row + 0.5) * step
            for col in range(self.width):
                lon = west + (col + 0.5) * step
                cells[row, col] = h3.latlng_to_cell(lat, lon, resolution)
        self._cell_index[resolution] = cells
        return cells

    def rasterise(
        self,
        values_by_cell: dict[str, int],
        resolution: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Paint cell values onto the grid (0 = background).

        Args:
            values_by_cell (dict[str, int]): H3 index to pixel value.
            resolution (int): H3 resolution of the supplied cells.
            out (Optional[np.ndarray]): Accumulate into an existing grid.

        Returns:
            np.ndarray: (H, W) uint32 grid.
        """
        grid = (
            out if out is not None
            else np.zeros(self.shape, dtype=np.uint32)
        )
        assert grid.shape == self.shape, "accumulator shape mismatch"
        pixel_cells = self._pixel_cells(resolution)
        flat_cells = pixel_cells.ravel()
        flat_grid = grid.ravel()
        for idx in range(flat_cells.size):
            value = values_by_cell.get(flat_cells[idx])
            if value is not None:
                flat_grid[idx] = value
        return grid


@register(WK_LAYER_BUILDERS, "regions")
def _regions_layer(
    ctx: ExportContext, raster: HexRasteriser, cfg: WebknossosConfig,
) -> LayerSpec:
    """Segmentation layer: one label per region, in sorted-name order."""
    values: dict[str, int] = {}
    for label, (_, cells) in enumerate(
        sorted(ctx.region_cells.items()), start=1,
    ):
        for cell in cells:
            values[cell] = label
    grid = raster.rasterise(values, ctx.base_resolution)
    return LayerSpec(
        "regions", grid, SEGMENTATION_CATEGORY,
        largest_segment_id=len(ctx.region_cells),
    )


@register(WK_LAYER_BUILDERS, "overlaps")
def _overlaps_layer(
    ctx: ExportContext, raster: HexRasteriser, cfg: WebknossosConfig,
) -> LayerSpec:
    """Segmentation layer: one label per intersecting region pair."""
    values: dict[str, int] = {}
    for label, (_, cells) in enumerate(
        sorted(ctx.overlap_cells.items()), start=1,
    ):
        for cell in cells:
            values[cell] = label
    grid = raster.rasterise(values, ctx.base_resolution)
    return LayerSpec(
        "overlaps", grid, SEGMENTATION_CATEGORY,
        largest_segment_id=max(len(ctx.overlap_cells), 1),
    )


@register(WK_LAYER_BUILDERS, "kml_clusters")
def _clusters_layer(
    ctx: ExportContext, raster: HexRasteriser, cfg: WebknossosConfig,
) -> LayerSpec:
    """Segmentation layer: cluster_id + 1, rasterised per resolution."""
    grid = np.zeros(raster.shape, dtype=np.uint32)
    for cluster_id, (resolution, cells) in sorted(
        ctx.cluster_cells.items(),
    ):
        values = {cell: cluster_id + 1 for cell in cells}
        raster.rasterise(values, resolution, out=grid)
    return LayerSpec(
        "kml_clusters", grid, SEGMENTATION_CATEGORY,
        largest_segment_id=max(
            (cid + 1 for cid in ctx.cluster_cells), default=1,
        ),
    )


@register(WK_LAYER_BUILDERS, "bias_heat")
def _bias_heat_layer(
    ctx: ExportContext, raster: HexRasteriser, cfg: WebknossosConfig,
) -> LayerSpec:
    """Colour layer: per-region SRI painted as a uint8 heat map."""
    values: dict[str, int] = {}
    for name, cells in ctx.region_cells.items():
        sri = float(ctx.bias_report.get(name, {}).get("sri_score", 0.0))
        assert 0.0 <= sri <= 1.0, "sri_score out of range"
        intensity = max(int(round(sri * cfg.heat_scale)), 1)
        for cell in cells:
            values[cell] = intensity
    grid = raster.rasterise(values, ctx.base_resolution)
    return LayerSpec(
        "bias_heat", grid.astype(np.uint8), COLOR_CATEGORY,
    )


def _footprint_to_cells(geometry: dict, resolution: int) -> set[str]:
    """Convert a GeoJSON Polygon footprint to H3 cells (outer ring only).

    Args:
        geometry (dict): GeoJSON geometry in (lon, lat) order.
        resolution (int): Target H3 resolution.

    Returns:
        set[str]: Covering cells; empty for non-Polygon or bad input.
    """
    if geometry.get("type") != "Polygon":
        return set()
    rings = geometry.get("coordinates", [])
    if not rings or len(rings[0]) < 4:
        return set()
    outer = [(float(lat), float(lon)) for lon, lat in rings[0]]
    poly = h3.LatLngPoly(outer)
    return set(h3.polygon_to_cells(poly, resolution))


@register(WK_LAYER_BUILDERS, "scene_coverage")
def _scene_coverage_layer(
    ctx: ExportContext, raster: HexRasteriser, cfg: WebknossosConfig,
) -> LayerSpec:
    """Colour layer: per-cell count of EO scene footprints covering it."""
    counts: dict[str, int] = {}
    for geometry in ctx.eo_footprints:
        for cell in _footprint_to_cells(geometry, ctx.base_resolution):
            counts[cell] = counts.get(cell, 0) + 1
    grid = raster.rasterise(counts, ctx.base_resolution)
    clipped = np.clip(grid, 0, cfg.heat_scale).astype(np.uint8)
    return LayerSpec("scene_coverage", clipped, COLOR_CATEGORY)


def _default_dataset_factory(
    path: Path, voxel_size: tuple[float, ...],
) -> Any:
    """Create a real ``webknossos.Dataset`` (lazy import).

    Args:
        path (Path): Dataset directory.
        voxel_size (tuple[float, ...]): Physical voxel size (nm).

    Returns:
        Any: An open ``webknossos.Dataset``.

    Raises:
        ImportError: If the ``wk`` extra is not installed.
    """
    try:
        import webknossos as wk
    except ImportError as exc:
        raise ImportError(
            "WEBKNOSSOS export requires the 'wk' extra:"
            " uv pip install 'hexcover[wk]'"
        ) from exc
    return wk.Dataset(path, voxel_size=voxel_size, exist_ok=True)


class WebknossosExporter:
    """Build, write, and optionally publish a WEBKNOSSOS dataset."""

    def __init__(
        self,
        config: WebknossosConfig,
        limits: LimitsConfig,
        dataset_factory: Callable[..., Any] = _default_dataset_factory,
    ) -> None:
        """Validate the selected layers and store collaborators.

        Args:
            config (WebknossosConfig): Export settings.
            limits (LimitsConfig): Shared hard limits.
            dataset_factory (Callable[..., Any]): Injection point for
                tests; defaults to the real ``webknossos.Dataset``.

        Raises:
            KeyError: If a selected layer is not registered.
        """
        unknown = set(config.layers) - set(WK_LAYER_BUILDERS)
        if unknown:
            raise KeyError(
                f"unknown webknossos layers {sorted(unknown)};"
                f" available: {sorted(WK_LAYER_BUILDERS)}"
            )
        self._cfg = config
        self._limits = limits
        self._factory = dataset_factory

    def export(
        self, ctx: ExportContext, bbox: tuple[float, float, float, float],
    ) -> Path:
        """Rasterise every selected layer and write the dataset.

        Args:
            ctx (ExportContext): Finished pipeline state.
            bbox (tuple[float, float, float, float]): (S, W, N, E)
                raster bounds, padding already applied.

        Returns:
            Path: The written dataset directory.
        """
        raster = HexRasteriser(
            bbox, self._cfg.voxel_deg, self._cfg.max_raster_pixels,
        )
        print(
            f"  raster grid {raster.width}x{raster.height}"
            f" at {self._cfg.voxel_deg} deg/voxel"
        )
        dataset_path = Path(self._cfg.dataset_dir) / self._cfg.dataset_name
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = self._factory(
            dataset_path, tuple(self._cfg.voxel_size_nm),
        )

        for name in self._cfg.layers:
            spec = WK_LAYER_BUILDERS[name](ctx, raster, self._cfg)
            self._write_layer(dataset, spec)

        self._publish_local(dataset_path)
        self._publish_remote(dataset)
        return dataset_path

    def _write_layer(self, dataset: Any, spec: LayerSpec) -> None:
        """Write one layer at mag 1 and optionally build the pyramid."""
        assert spec.data.ndim == 2, "layers must be 2D grids"
        kwargs: dict[str, Any] = {
            "category": spec.category,
            "dtype_per_channel": str(spec.data.dtype),
        }
        if spec.category == SEGMENTATION_CATEGORY:
            kwargs["largest_segment_id"] = spec.largest_segment_id
        layer = dataset.add_layer(spec.name, **kwargs)
        volume = spec.data.T[:, :, np.newaxis]
        layer.add_mag(1).write(np.ascontiguousarray(volume))
        if self._cfg.downsample:
            layer.downsample()
        occupied = int(np.count_nonzero(spec.data))
        print(
            f"  layer '{spec.name}' ({spec.category}):"
            f" {occupied} occupied voxels"
        )

    def _publish_local(self, dataset_path: Path) -> None:
        """Copy the dataset into a local instance's binaryData folder."""
        if not self._cfg.binary_data_dir:
            return
        target = Path(self._cfg.binary_data_dir) / self._cfg.dataset_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(dataset_path, target)
        print(f"  copied dataset into local binaryData: {target}")

    def _publish_remote(self, dataset: Any) -> None:
        """Upload to a remote or local WEBKNOSSOS server if configured."""
        if not self._cfg.upload:
            return
        token = os.environ.get(self._cfg.token_env, "")
        if not token:
            print(
                f"  upload skipped: set {self._cfg.token_env}"
                " in the environment"
            )
            return
        import webknossos as wk
        with wk.webknossos_context(url=self._cfg.url, token=token):
            remote = dataset.upload()
        print(f"  uploaded to {self._cfg.url}: {remote.url}")