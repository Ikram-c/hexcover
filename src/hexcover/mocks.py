"""Deterministic synthetic data and offline stand-ins for external services."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Union

import numpy as np
import requests

from .config import (
    BiasConfig,
    ClusterConfig,
    Config,
    EOConfig,
    GJKConfig,
    HierarchyConfig,
    HullConfig,
    KMLConfig,
    LimitsConfig,
    OutputPaths,
    OverlapSettings,
    OverpassConfig,
    PipelineSettings,
    PrecisionConfig,
    RegionConfig,
    WebknossosConfig,
)
from .eo_client import SatelliteScene


@dataclass(frozen=True)
class MockDataConfig:
    """Tuning knobs for deterministic synthetic geometry."""

    seed: int = 42
    ring_points: int = 64
    ring_noise_deg: float = 0.05
    points_per_cluster: int = 12
    cluster_spread_deg: float = 0.25


def synthetic_ring(
    centre: tuple[float, float],
    radius_deg: float,
    cfg: MockDataConfig = MockDataConfig(),
) -> np.ndarray:
    """Generate a noisy ring of points around a centre.

    Args:
        centre (tuple[float, float]): (lat, lon) ring centre.
        radius_deg (float): Ring radius in degrees.
        cfg (MockDataConfig): Determinism and noise settings.

    Returns:
        np.ndarray: (N, 2) (lat, lon) array.
    """
    assert radius_deg > 0, "radius must be positive"
    assert cfg.ring_points >= 3, "a ring needs at least three points"
    rng = np.random.default_rng(cfg.seed)
    angles = np.linspace(0.0, 2.0 * np.pi, cfg.ring_points, endpoint=False)
    lat = centre[0] + radius_deg * np.sin(angles)
    lon = centre[1] + radius_deg * np.cos(angles)
    noise = rng.normal(0.0, cfg.ring_noise_deg, (cfg.ring_points, 2))
    return np.column_stack([lat, lon]) + noise


def synthetic_clusters(
    centres: Iterable[tuple[float, float]],
    cfg: MockDataConfig = MockDataConfig(),
) -> np.ndarray:
    """Generate Gaussian point blobs around each centre.

    Args:
        centres (Iterable[tuple[float, float]]): (lat, lon) blob centres.
        cfg (MockDataConfig): Determinism and spread settings.

    Returns:
        np.ndarray: Stacked (N, 2) (lat, lon) array.
    """
    assert cfg.points_per_cluster > 0, "clusters need at least one point"
    blobs: list[np.ndarray] = []
    for idx, centre in enumerate(centres):
        rng = np.random.default_rng(cfg.seed + idx)
        blob = rng.normal(
            loc=centre,
            scale=cfg.cluster_spread_deg,
            size=(cfg.points_per_cluster, 2),
        )
        blobs.append(blob)
    assert len(blobs) > 0, "at least one centre required"
    return np.vstack(blobs)


def overpass_payload(points: np.ndarray) -> dict:
    """Build a raw Overpass JSON payload from a point array.

    Args:
        points (np.ndarray): (N, 2) (lat, lon) array.

    Returns:
        dict: Payload matching the Overpass ``out body`` shape.
    """
    assert points.ndim == 2 and points.shape[1] == 2, "expected (N, 2) input"
    return {
        "elements": [
            {"type": "node", "lat": float(lat), "lon": float(lon)}
            for lat, lon in points
        ]
    }


def kml_bytes(points: np.ndarray) -> bytes:
    """Serialise points as a KML 2.2 document.

    Args:
        points (np.ndarray): (N, 2) (lat, lon) array.

    Returns:
        bytes: UTF-8 encoded KML document.
    """
    assert points.ndim == 2 and points.shape[1] == 2, "expected (N, 2) input"
    placemarks = "".join(
        "<Placemark><Point><coordinates>"
        f"{float(lon)!r},{float(lat)!r},0"
        "</coordinates></Point></Placemark>"
        for lat, lon in points
    )
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2">'
        f"<Document>{placemarks}</Document></kml>"
    )
    return doc.encode("utf-8")


def kmz_bytes(points: np.ndarray) -> bytes:
    """Serialise points as a KMZ archive containing ``doc.kml``.

    Args:
        points (np.ndarray): (N, 2) (lat, lon) array.

    Returns:
        bytes: ZIP-compressed KMZ payload.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", kml_bytes(points))
    return buffer.getvalue()


def synthetic_scenes(
    collection: str,
    target_date: datetime,
    count: int = 3,
    centre: tuple[float, float] = (55.0, 3.0),
) -> list[SatelliteScene]:
    """Generate deterministic scenes at hourly offsets around a date.

    Args:
        collection (str): Collection label for every scene.
        target_date (datetime): Centre of the acquisition spread.
        count (int): Number of scenes.
        centre (tuple[float, float]): (lat, lon) footprint centre.

    Returns:
        list[SatelliteScene]: Scenes at +1h, -2h, +3h, ... offsets.
    """
    assert count > 0, "count must be positive"
    scenes: list[SatelliteScene] = []
    lat, lon = centre
    footprint = {
        "type": "Polygon",
        "coordinates": [[
            [lon - 0.5, lat - 0.5], [lon + 0.5, lat - 0.5],
            [lon + 0.5, lat + 0.5], [lon - 0.5, lat + 0.5],
            [lon - 0.5, lat - 0.5],
        ]],
    }
    for i in range(count):
        offset = timedelta(hours=(i + 1) * (1 if i % 2 == 0 else -1))
        scenes.append(SatelliteScene(
            id=f"{collection}-{i:04d}",
            collection=collection,
            acquired=target_date + offset,
            cloud_cover=float(10 * i),
            geometry=footprint,
            download_url=f"https://example.test/{collection}/{i}",
        ))
    return scenes


class MockOverpassClient:
    """Offline drop-in for ``OverpassClient`` backed by fixed point arrays."""

    def __init__(self, relations: Mapping[int, np.ndarray]) -> None:
        """Store the relation-id to point-array mapping.

        Args:
            relations (Mapping[int, np.ndarray]): Points per relation ID.
        """
        assert len(relations) > 0, "at least one relation required"
        self._relations = {int(key): val for key, val in relations.items()}
        self.calls: list[int] = []

    def fetch_relation_nodes(self, relation_id: int) -> np.ndarray:
        """Return the stored points for ``relation_id``.

        Args:
            relation_id (int): OSM relation ID.

        Returns:
            np.ndarray: Copy of the stored (N, 2) array.

        Raises:
            KeyError: If the relation ID has no stored points.
        """
        assert relation_id > 0, "relation_id must be positive"
        self.calls.append(relation_id)
        if relation_id not in self._relations:
            raise KeyError(f"no mock data for relation {relation_id}")
        return self._relations[relation_id].copy()

    @classmethod
    def from_regions(
        cls,
        regions: Iterable[RegionConfig],
        base_centre: tuple[float, float] = (55.0, 3.0),
        spacing_deg: float = 1.5,
        radius_deg: float = 2.0,
        cfg: MockDataConfig = MockDataConfig(),
    ) -> "MockOverpassClient":
        """Generate one deterministic ring per region along a lon axis.

        With ``spacing_deg < 2 * radius_deg`` adjacent region hulls
        overlap, which exercises the GJK stage.

        Args:
            regions (Iterable[RegionConfig]): Regions to fabricate.
            base_centre (tuple[float, float]): First ring centre.
            spacing_deg (float): Longitude step between ring centres.
            radius_deg (float): Ring radius in degrees.
            cfg (MockDataConfig): Determinism settings.

        Returns:
            MockOverpassClient: Client seeded with one ring per region.
        """
        relations: dict[int, np.ndarray] = {}
        for idx, region in enumerate(regions):
            centre = (base_centre[0], base_centre[1] + idx * spacing_deg)
            ring_cfg = replace(cfg, seed=cfg.seed + idx)
            relations[region.relation_id] = synthetic_ring(
                centre, radius_deg, ring_cfg,
            )
        return cls(relations)


class MockEOClient:
    """Offline drop-in for ``EOClient`` backed by canned scene lists."""

    def __init__(
        self, scenes_by_collection: Mapping[str, list[SatelliteScene]],
    ) -> None:
        """Store the collection-to-scenes mapping.

        Args:
            scenes_by_collection (Mapping[str, list[SatelliteScene]]):
                Scenes returned per collection key.
        """
        self._scenes = dict(scenes_by_collection)
        self.calls: list[dict] = []

    def iter_scenes(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Iterator[SatelliteScene]:
        """Record the call and yield the canned scenes (limit-capped)."""
        self.calls.append({
            "collection": collection, "bbox": bbox,
            "start": start_date, "end": end_date, "limit": limit,
        })
        scenes = self._scenes.get(collection, [])
        cap = limit if limit is not None else len(scenes)
        yield from scenes[:cap]


class MockZarrGroup:
    """Dict-backed stand-in for a read-only Zarr group."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        attrs: Optional[dict] = None,
        groups: Optional[Mapping[str, "MockZarrGroup"]] = None,
    ) -> None:
        """Store arrays, attributes, and optional nested groups.

        Args:
            arrays (Mapping[str, np.ndarray]): Named arrays.
            attrs (Optional[dict]): Group attributes.
            groups (Optional[Mapping[str, MockZarrGroup]]): Sub-groups.
        """
        self._arrays = {key: np.asarray(val) for key, val in arrays.items()}
        self._groups = dict(groups or {})
        self.attrs = dict(attrs or {})

    def __getitem__(
        self, key: str,
    ) -> Union[np.ndarray, "MockZarrGroup"]:
        """Return the named sub-group or array.

        Raises:
            KeyError: If the key names neither.
        """
        if key in self._groups:
            return self._groups[key]
        if key in self._arrays:
            return self._arrays[key]
        raise KeyError(key)


class MockResponse:
    """Minimal ``requests.Response`` stand-in."""

    def __init__(
        self,
        content: bytes = b"",
        status_code: int = 200,
        json_payload: Optional[dict] = None,
    ) -> None:
        """Store the canned body, status code, and optional JSON."""
        self.content = content
        self.status_code = status_code
        self.reason = "OK" if status_code < 400 else "ERROR"
        self._json = json_payload

    def json(self) -> dict:
        """Return the canned JSON payload.

        Raises:
            AssertionError: If no payload was scripted.
        """
        assert self._json is not None, "no JSON payload scripted"
        return self._json

    def raise_for_status(self) -> None:
        """Raise ``requests.HTTPError`` for 4xx/5xx statuses.

        Raises:
            requests.HTTPError: If the status code is >= 400.
        """
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} {self.reason}", response=self,
            )


class MockSession:
    """Scripted ``requests.Session`` stand-in returning canned responses."""

    def __init__(self, responses: Iterable[MockResponse]) -> None:
        """Store the response script in call order.

        Args:
            responses (Iterable[MockResponse]): One per expected call.
        """
        self._responses = list(responses)
        assert len(self._responses) > 0, "at least one response required"
        self.requests: list[dict[str, Any]] = []
        self.headers: dict = {}

    def get(
        self,
        url: str,
        timeout: Optional[int] = None,
        headers: Optional[dict] = None,
    ) -> MockResponse:
        """Record the call and return the next scripted response.

        Raises:
            AssertionError: If called more times than scripted.
        """
        self.requests.append(
            {"url": url, "timeout": timeout, "headers": headers},
        )
        assert len(self.requests) <= len(self._responses), (
            "MockSession called more times than scripted"
        )
        return self._responses[len(self.requests) - 1]

    def post(
        self,
        url: str,
        data: Optional[dict] = None,
        json: Optional[dict] = None,
        timeout: Optional[int] = None,
        headers: Optional[dict] = None,
    ) -> MockResponse:
        """Record the call and return the next scripted response.

        Raises:
            AssertionError: If called more times than scripted.
        """
        self.requests.append(
            {"url": url, "data": data, "json": json, "timeout": timeout},
        )
        assert len(self.requests) <= len(self._responses), (
            "MockSession called more times than scripted"
        )
        return self._responses[len(self.requests) - 1]


class MockWKMag:
    """Capture the array written to one mag level."""

    def __init__(self) -> None:
        """Initialise with no data written."""
        self.data: Optional[np.ndarray] = None

    def write(self, data: np.ndarray) -> None:
        """Store a copy of the written volume."""
        self.data = np.array(data)


class MockWKLayer:
    """Record layer metadata, writes, and downsample calls."""

    def __init__(self, name: str, kwargs: dict) -> None:
        """Store the layer name and creation kwargs."""
        self.name = name
        self.kwargs = kwargs
        self.mags: dict[int, MockWKMag] = {}
        self.downsampled = False

    def add_mag(self, mag: int) -> MockWKMag:
        """Create and return a mock mag level."""
        self.mags[mag] = MockWKMag()
        return self.mags[mag]

    def downsample(self) -> None:
        """Record that a pyramid build was requested."""
        self.downsampled = True


class MockWKDataset:
    """Offline drop-in for ``webknossos.Dataset``."""

    def __init__(
        self, path: Any, voxel_size: tuple[float, ...],
    ) -> None:
        """Record the dataset path and voxel size."""
        self.path = Path(path)
        self.voxel_size = voxel_size
        self.layers: dict[str, MockWKLayer] = {}
        self.uploaded = False
        self.path.mkdir(parents=True, exist_ok=True)

    def add_layer(self, name: str, **kwargs: Any) -> MockWKLayer:
        """Create, register, and return a mock layer.

        Raises:
            AssertionError: On duplicate layer names.
        """
        assert name not in self.layers, f"duplicate layer: {name}"
        self.layers[name] = MockWKLayer(name, kwargs)
        return self.layers[name]

    def upload(self) -> "MockWKDataset":
        """Mark the dataset as uploaded and return self."""
        self.uploaded = True
        return self


def mock_wk_dataset_factory(
    created: list[MockWKDataset],
) -> Callable[..., MockWKDataset]:
    """Return a dataset factory that records every dataset it creates.

    Args:
        created (list[MockWKDataset]): Sink for created datasets.

    Returns:
        Callable[..., MockWKDataset]: Factory for ``WebknossosExporter``.
    """
    def _factory(
        path: Any, voxel_size: tuple[float, ...],
    ) -> MockWKDataset:
        dataset = MockWKDataset(path, voxel_size)
        created.append(dataset)
        return dataset
    return _factory


def demo_config(
    base_dir: Union[str, Path],
    regions: Iterable[RegionConfig] = (),
    kml_enabled: bool = False,
    h3_resolution: int = 3,
    eo: Optional[EOConfig] = None,
    webknossos: Optional[WebknossosConfig] = None,
) -> Config:
    """Build an offline-friendly ``Config`` rooted under ``base_dir``.

    Retries and rate limiting are zeroed so failures surface instantly
    and mocked runs never sleep. EO and WEBKNOSSOS default to disabled;
    pass configs to enable them.

    Args:
        base_dir (Union[str, Path]): Root for all output paths.
        regions (Iterable[RegionConfig]): Regions to process.
        kml_enabled (bool): Enable the KML stage (inject a source).
        h3_resolution (int): Base H3 resolution.
        eo (Optional[EOConfig]): EO query settings.
        webknossos (Optional[WebknossosConfig]): Export settings.

    Returns:
        Config: Fully populated configuration.
    """
    out = str(Path(base_dir))
    output = OutputPaths(
        regions_geojson=f"{out}/region_hexagons.geojson",
        combined_geojson=f"{out}/combined_hexagons.geojson",
        overlaps_geojson=f"{out}/overlap_hexagons.geojson",
        bias_report=f"{out}/geo_bias_report.json",
        kml_hulls_geojson=f"{out}/kml_hulls.geojson",
        kml_hexagons_geojson=f"{out}/kml_cluster_hexagons.geojson",
        eo_scenes_csv=f"{out}/eo_scenes.csv",
        eo_footprints_geojson=f"{out}/eo_scene_footprints.geojson",
    )
    return Config(
        pipeline=PipelineSettings(h3_resolution=h3_resolution),
        regions=tuple(regions),
        overlap=OverlapSettings(),
        limits=LimitsConfig(),
        overpass=OverpassConfig(max_retries=0, rate_limit_s=0),
        hull=HullConfig(),
        gjk=GJKConfig(),
        bias=BiasConfig(
            features=("sri_entropy", "gini", "occupancy", "dispersion"),
        ),
        cluster=ClusterConfig(),
        hierarchy=HierarchyConfig(),
        precision=PrecisionConfig(),
        output=output,
        kml=KMLConfig(enabled=kml_enabled),
        eo=eo or EOConfig(),
        webknossos=webknossos or WebknossosConfig(),
    )