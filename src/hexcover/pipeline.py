"""End-to-end orchestration and CLI entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import h3
import numpy as np
import requests

from .bias import BiasScorer
from .clustering import PointClusterer, ResolutionMapper
from .config import Config, RegionConfig
from .decorators import stage
from .eo_client import EOClient, build_eo_client
from .eo_query import (
    query_availability,
    scene_footprint_features,
    strip_private_fields,
)
from .geometry import HexConverter, convex_hull, hull_area_sq_deg
from .kml_sources import KMLPointSource, build_kml_source
from .overlap import GJKTester, OverlapDetector
from .overpass import OverpassClient
from .utils import AtomicWriter
from .wk_export import ExportContext, WebknossosExporter


class HexCoverPipeline:
    """Compose Overpass fetching, hulls, hex covering, clustering, GJK,
    EO availability, and WEBKNOSSOS export."""

    def __init__(
        self,
        config: Config,
        kml_source: Optional[KMLPointSource] = None,
        overpass_client: Optional[OverpassClient] = None,
        eo_client: Optional[EOClient] = None,
        wk_exporter: Optional[WebknossosExporter] = None,
    ) -> None:
        """Initialise every subcomponent from the supplied configuration.

        Args:
            config (Config): Full pipeline configuration.
            kml_source (Optional[KMLPointSource]): Injected source override.
            overpass_client (Optional[OverpassClient]): Injected client
                override; pass ``MockOverpassClient`` for offline runs.
            eo_client (Optional[EOClient]): Injected EO client override;
                pass ``MockEOClient`` for offline runs.
            wk_exporter (Optional[WebknossosExporter]): Injected exporter
                override for testing or custom dataset factories.
        """
        self._cfg = config
        self._hex = HexConverter(
            config.pipeline.h3_resolution, config.hull.min_vertices,
        )
        self._overpass = (
            overpass_client
            if overpass_client is not None
            else OverpassClient(config.overpass)
        )
        self._bias = BiasScorer(config.bias, config.limits)
        self._clusterer = PointClusterer(config.cluster, config.limits)
        self._mapper = ResolutionMapper(config.hierarchy)
        self._overlaps = OverlapDetector(
            GJKTester(config.gjk), config.precision,
        )
        self._writer = AtomicWriter()
        self._kml_source = kml_source
        self._eo_client = eo_client
        self._wk = (
            wk_exporter
            if wk_exporter is not None
            else (
                WebknossosExporter(config.webknossos, config.limits)
                if config.webknossos.enabled
                else None
            )
        )

        self._region_hulls: dict[str, np.ndarray] = {}
        self._region_cells: dict[str, set[str]] = {}
        self._region_type_map: dict[str, str] = {}
        self._overlap_names: list[str] = []
        self._overlap_cells: dict[str, set[str]] = {}
        self._cluster_cells: dict[int, tuple[int, set[str]]] = {}
        self._eo_footprints: list[dict] = []
        self._all_features: list[dict] = []
        self._bias_report: dict[str, Any] = {}
        self._overlap_pairs: list[dict[str, Any]] = []

    def run(self) -> None:
        """Run every pipeline stage and print a final summary."""
        self._process_regions()
        self._process_kml_clusters()
        self._detect_overlaps()
        self._writer.write_features(
            self._all_features, self._cfg.output.combined_geojson,
        )
        self._write_bias_report()
        self._query_eo_scenes()
        self._export_webknossos()
        print(
            f"\nDone: {len(self._all_features)} total hex features"
            f" at base resolution {self._cfg.pipeline.h3_resolution}."
        )

    @stage("Regions")
    def _process_regions(self) -> None:
        """Fetch, hull, and hex-cover every configured region."""
        features: list[dict] = []
        for region in self._cfg.regions:
            print(f"\n  {region.name} (relation {region.relation_id})")
            features.extend(self._process_region(region))
        self._writer.write_features(
            features, self._cfg.output.regions_geojson,
        )
        self._all_features.extend(features)

    def _process_region(self, region: RegionConfig) -> list[dict]:
        """Process one region and return its hex features."""
        try:
            raw = self._overpass.fetch_relation_nodes(region.relation_id)
        except (requests.RequestException, RuntimeError) as exc:
            print(f"  failed after retries: {exc}")
            return []

        bbox = self._cfg.pipeline.clip_bbox
        if region.clip_to_bbox and bbox is not None:
            raw = self._clip(raw, bbox)
            print(f"    clipped to {len(raw)} points")
        if len(raw) < self._cfg.hull.min_vertices:
            print("    too few points, skipping")
            return []

        hull = convex_hull(raw, self._cfg.hull, self._cfg.limits)
        cells = self._hex.hull_to_cells(hull)
        print(f"    hull: {len(hull)} vertices, {len(cells)} H3 cells")

        self._region_hulls[region.name] = hull
        self._region_cells[region.name] = cells
        self._region_type_map[region.name] = region.type
        if self._in_overlap_set(region):
            self._overlap_names.append(region.name)
        self._score_bias(region.name, cells)
        return list(self._hex.iter_features(
            cells, {"name": region.name, "type": region.type},
        ))

    def _in_overlap_set(self, region: RegionConfig) -> bool:
        """Return True if the region participates in overlap detection."""
        if not self._cfg.overlap.enabled:
            return False
        allowed = self._cfg.overlap.region_types
        return not allowed or region.type in allowed

    @staticmethod
    def _clip(pts: np.ndarray, bbox: tuple[float, ...]) -> np.ndarray:
        """Keep only points inside the supplied (S, W, N, E) bbox."""
        assert len(bbox) == 4, "bbox must have exactly four values"
        south, west, north, east = bbox
        mask = (
            (pts[:, 0] >= south) & (pts[:, 0] <= north)
            & (pts[:, 1] >= west) & (pts[:, 1] <= east)
        )
        return pts[mask]

    @stage("KML cluster hierarchical hexagons")
    def _process_kml_clusters(self) -> None:
        """Cluster KML points, hull each cluster, and hex-cover by area."""
        if not self._cfg.kml.enabled:
            print("  disabled")
            return
        source = self._kml_source or build_kml_source(self._cfg.kml.source)
        points = source.parse_points()
        if len(points) == 0:
            print("  no KML points found")
            return
        print(f"  parsed {len(points)} KML points")

        clusters = self._clusterer.cluster(points)
        print(f"  formed {len(clusters)} cluster(s)")
        hull_features: list[dict] = []
        hex_features: list[dict] = []
        for idx, cluster_pts in enumerate(clusters):
            self._cover_cluster(idx, cluster_pts, hull_features, hex_features)

        if hull_features:
            self._writer.write_features(
                hull_features, self._cfg.output.kml_hulls_geojson,
            )
        if hex_features:
            self._writer.write_features(
                hex_features, self._cfg.output.kml_hexagons_geojson,
            )
            self._all_features.extend(hex_features)

    def _cover_cluster(
        self,
        idx: int,
        cluster_pts: np.ndarray,
        hull_features: list[dict],
        hex_features: list[dict],
    ) -> None:
        """Hull one cluster, pick a resolution by area, and cover it."""
        if len(cluster_pts) < 3:
            print(f"    cluster {idx}: skipped (insufficient points)")
            return
        hull = convex_hull(cluster_pts, self._cfg.hull, self._cfg.limits)
        if len(hull) < 3:
            print(f"    cluster {idx}: skipped (insufficient hull)")
            return

        area = hull_area_sq_deg(hull)
        res = self._mapper.resolution_for_area(area)
        cells = self._hex.hull_to_cells(hull, resolution=res)
        self._cluster_cells[idx] = (int(res), cells)
        props = {
            "cluster_id": idx,
            "n_points": int(len(cluster_pts)),
            "hull_area_sq_deg": round(
                area, self._cfg.precision.area_decimal_places,
            ),
            "h3_resolution": int(res),
            "type": "kml_cluster",
        }
        print(
            f"    cluster {idx}: {len(cluster_pts)} pts,"
            f" area={area:.4f} sq deg, res={res}, cells={len(cells)}"
        )
        hull_features.append(self._hex.hull_to_polygon_feature(hull, props))
        hex_features.extend(self._hex.iter_features(cells, props))
        self._score_bias(f"kml_cluster_{idx}", cells)

    @stage("Overlap detection")
    def _detect_overlaps(self) -> None:
        """Find bbox-overlap components and run GJK per pair."""
        if not self._cfg.overlap.enabled or len(self._overlap_names) < 2:
            print("  skipped (disabled or fewer than two regions)")
            return
        bboxes = [
            self._overlaps.bbox_of_hull(self._region_hulls[name])
            for name in self._overlap_names
        ]
        groups = self._overlaps.find_groups(bboxes)
        print(f"  {len(groups)} bounding-box overlap group(s)")
        self._overlap_pairs = self._overlaps.test_pairs(
            groups, self._overlap_names, self._region_hulls,
        )
        features = list(self._iter_overlap_features())
        if features:
            self._writer.write_features(
                features, self._cfg.output.overlaps_geojson,
            )
        else:
            print("  no polygon-level overlaps to write")

    def _iter_overlap_features(self) -> Iterator[dict]:
        """Yield hex features for every intersecting region pair."""
        for pair in self._overlap_pairs:
            if not pair["intersects"]:
                continue
            shared = (
                self._region_cells.get(pair["a"], set())
                & self._region_cells.get(pair["b"], set())
            )
            self._overlap_cells[f"{pair['a']} / {pair['b']}"] = shared
            yield from self._hex.iter_features(
                shared,
                {"name": f"{pair['a']} / {pair['b']}", "type": "overlap"},
            )

    def _score_bias(self, region_name: str, cells: set[str]) -> None:
        """Score one region's cells if bias reporting is enabled."""
        if self._cfg.bias.enabled:
            self._bias_report[region_name] = self._bias.score(cells)

    def _write_bias_report(self) -> None:
        """Emit the spatial bias JSON report and a stdout summary."""
        if not self._cfg.bias.enabled:
            return
        self._bias_report["_overlap_pairs"] = self._overlap_pairs
        self._writer.write_json(
            self._bias_report, self._cfg.output.bias_report,
        )
        print(
            f"\nSpatial bias summary"
            f" (features: {', '.join(self._cfg.bias.features)})"
        )
        for region, data in self._bias_report.items():
            if region.startswith("_"):
                continue
            metrics = ", ".join(
                f"{key}={val}"
                for key, val in data.items()
                if key not in ("grid_counts", "n_cells")
            )
            print(f"  {region:20s}  {metrics}  ({data['n_cells']} cells)")

    @stage("EO scene availability")
    def _query_eo_scenes(self) -> None:
        """Query satellite scene availability around each target centroid."""
        if not self._cfg.eo.enabled:
            print("  disabled")
            return
        unknown = set(self._cfg.eo.targets) - {"regions", "kml_clusters"}
        assert not unknown, f"unknown eo.targets: {sorted(unknown)}"

        client = self._eo_client or build_eo_client(self._cfg.eo)
        reference = self._eo_reference_date()
        print(f"  reference date: {reference.isoformat()}")

        records: list[dict] = []
        for name, lat, lon in self._iter_eo_targets():
            rows = query_availability(
                lat=lat, lon=lon, target_date=reference,
                client=client, cfg=self._cfg.eo,
                limit=self._cfg.eo.batch_result_limit,
                extra={"target": name, "lat": lat, "lon": lon},
            )
            print(f"  {name}: {len(rows)} scene(s)")
            records.extend(rows)

        self._eo_footprints = [
            r["_geometry"] for r in records if r.get("_geometry")
        ]

        if not records:
            print("  no scenes found")
            return
        self._writer.write_csv(
            strip_private_fields(records), self._cfg.output.eo_scenes_csv,
        )
        footprints = list(scene_footprint_features(records))
        if footprints:
            self._writer.write_features(
                footprints, self._cfg.output.eo_footprints_geojson,
            )

    def _eo_reference_date(self) -> datetime:
        """Return the configured reference date, defaulting to now (UTC)."""
        raw = self._cfg.eo.reference_date
        if raw:
            return datetime.fromisoformat(raw)
        return datetime.now(timezone.utc).replace(tzinfo=None)

    def _iter_eo_targets(self) -> Iterator[tuple[str, float, float]]:
        """Yield ``(name, lat, lon)`` centroids for the configured targets."""
        targets = self._cfg.eo.targets
        if "regions" in targets:
            for name, hull in self._region_hulls.items():
                yield (
                    name,
                    float(hull[:, 0].mean()),
                    float(hull[:, 1].mean()),
                )
        if "kml_clusters" in targets:
            for idx, (_, cells) in sorted(self._cluster_cells.items()):
                pts = [h3.cell_to_latlng(cell) for cell in cells]
                assert pts, "cluster with no cells"
                lat = sum(p[0] for p in pts) / len(pts)
                lon = sum(p[1] for p in pts) / len(pts)
                yield (f"kml_cluster_{idx}", lat, lon)

    @stage("WEBKNOSSOS export")
    def _export_webknossos(self) -> None:
        """Rasterise results into a WEBKNOSSOS dataset for the GUI."""
        if self._wk is None:
            print("  disabled")
            return
        if not self._region_cells and not self._cluster_cells:
            print("  nothing to export")
            return
        ctx = ExportContext(
            region_cells=self._region_cells,
            region_types=self._region_type_map,
            overlap_cells=self._overlap_cells,
            cluster_cells=self._cluster_cells,
            bias_report=self._bias_report,
            base_resolution=self._cfg.pipeline.h3_resolution,
            eo_footprints=tuple(self._eo_footprints),
        )
        self._wk.export(ctx, self._raster_bbox())

    def _raster_bbox(self) -> tuple[float, float, float, float]:
        """Return padded (S, W, N, E) bounds covering every cell set."""
        lats: list[float] = []
        lons: list[float] = []
        cell_sets = list(self._region_cells.values()) + [
            cells for _, cells in self._cluster_cells.values()
        ]
        for cells in cell_sets:
            for cell in cells:
                lat, lon = h3.cell_to_latlng(cell)
                lats.append(lat)
                lons.append(lon)
        assert lats, "raster bbox requested with no cells"
        pad = self._cfg.webknossos.padding_deg
        return (
            min(lats) - pad, min(lons) - pad,
            max(lats) + pad, max(lons) + pad,
        )


def main() -> None:
    """Parse CLI arguments, load the config, and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Region / KML cluster H3 hexagon covering pipeline",
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    HexCoverPipeline(Config.from_yaml(args.config)).run()


if __name__ == "__main__":
    main()