"""Configuration dataclasses and generic YAML loader."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import yaml


def _from_mapping(cls: type, raw: dict) -> Any:
    """Hydrate a frozen dataclass from a YAML mapping, coercing lists.

    Args:
        cls (type): Dataclass type.
        raw (dict): Raw YAML mapping.

    Returns:
        Any: Populated dataclass instance.
    """
    assert isinstance(raw, dict), f"{cls.__name__} section must be a mapping"
    names = {f.name for f in fields(cls)}
    kwargs = {
        key: tuple(val) if isinstance(val, list) else val
        for key, val in raw.items()
        if key in names
    }
    return cls(**kwargs)


@dataclass(frozen=True)
class RegionConfig:
    """One OSM relation to fetch, hull, and hex-cover."""

    name: str
    relation_id: int
    type: str = "region"
    clip_to_bbox: bool = True


@dataclass(frozen=True)
class PipelineSettings:
    """Top-level pipeline identity settings."""

    h3_resolution: int = 3
    clip_bbox: Optional[tuple[float, ...]] = None


@dataclass(frozen=True)
class OverlapSettings:
    """Which regions participate in pairwise overlap detection."""

    enabled: bool = True
    region_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class LimitsConfig:
    """Hard upper bounds enforced by assertions (Power of 10 rule 2)."""

    max_input_points: int = 500_000
    max_cluster_points: int = 100_000
    max_hull_iterations_factor: int = 4
    max_bias_cells: int = 2_000_000


@dataclass(frozen=True)
class OverpassConfig:
    """Overpass API connection, retry, and identification settings."""

    urls: tuple[str, ...] = ("https://overpass-api.de/api/interpreter",)
    timeout_s: int = 180
    timeout_buffer_s: int = 30
    rate_limit_s: int = 15
    max_retries: int = 4
    backoff_base_s: float = 15.0
    backoff_multiplier: float = 2.0
    retryable_status_codes: tuple[int, ...] = (429, 502, 503, 504)
    user_agent: str = "hexcover/2.0"
    referer: str = "https://github.com/local/hexcover"


@dataclass(frozen=True)
class HullConfig:
    """Convex hull construction settings."""

    min_vertices: int = 3
    collinear_tolerance: float = 1.0e-10


@dataclass(frozen=True)
class GJKConfig:
    """Tuning knobs for the 2D GJK solver."""

    max_iterations: int = 128
    tolerance: float = 1.0e-6
    use_nesterov: bool = True
    norm_floor: float = 1.0e-30


@dataclass(frozen=True)
class BiasConfig:
    """Spatial-bias scorer settings, including the selected feature set."""

    enabled: bool = True
    features: tuple[str, ...] = ("sri_entropy",)
    grid_rows: int = 5
    grid_cols: int = 5
    degenerate_span: float = 1.0e-6
    decimal_places: int = 4


@dataclass(frozen=True)
class ClusterConfig:
    """Single-linkage clustering parameters."""

    distance_threshold_deg: float = 1.5
    min_cluster_size: int = 2
    top_n_clusters: Optional[int] = None
    pair_chunk_size: int = 512


@dataclass(frozen=True)
class HierarchyConfig:
    """Cluster hull area to H3 resolution mapping."""

    area_thresholds_sq_deg: tuple[float, ...] = (0.1, 1.0, 5.0, 20.0)
    resolutions: tuple[int, ...] = (7, 6, 5, 4, 3)


@dataclass(frozen=True)
class RetrySettings:
    """Generic bounded retry policy."""

    max_retries: int = 0
    backoff_base_s: float = 1.0
    backoff_multiplier: float = 2.0
    retryable_status_codes: tuple[int, ...] = (429, 502, 503, 504)


@dataclass(frozen=True)
class KMLSourceConfig:
    """YAML-buildable KML point source selector and parameters."""

    type: str = "file"
    path: str = "doc.kml"
    url: str = ""
    timeout_s: int = 30
    headers: dict = field(default_factory=dict)
    store: str = ""
    points_array: str = ""
    lat_array: str = "latitude"
    lon_array: str = "longitude"
    group_path: str = ""
    storage_options: dict = field(default_factory=dict)
    scale_index: int = 0
    apply_transforms: bool = True
    retry: RetrySettings = field(default_factory=RetrySettings)


@dataclass(frozen=True)
class KMLConfig:
    """Top-level KML cluster processing configuration."""

    enabled: bool = True
    source: KMLSourceConfig = field(default_factory=KMLSourceConfig)


@dataclass(frozen=True)
class EOProviderConfig:
    """Endpoint and identity settings for one EO catalogue provider."""

    base_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    stac_url: str = "https://catalogue.dataspace.copernicus.eu/stac"
    auth_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
        "/protocol/openid-connect/token"
    )
    client_id: str = "cdse-public"
    request_timeout_s: int = 30
    user_agent: str = "hexcover-eo/2.0"
    username_env: str = "CDSE_USERNAME"
    password_env: str = "CDSE_PASSWORD"


@dataclass(frozen=True)
class EOGeometryConfig:
    """Geometric and unit-conversion constants for bbox computation."""

    km_per_degree_lat: float = 111.0
    min_cos_clamp: float = 0.1
    seconds_per_hour: int = 3600


@dataclass(frozen=True)
class EOCSVColumns:
    """Column names expected in the batch input CSV."""

    date: str = "Start date"
    time: str = "Start time"
    latitude: str = "Start latitude"
    longitude: str = "Start longitude"
    id: str = "Id"
    camera_id: str = "CameraId"


@dataclass(frozen=True)
class EOConfig:
    """EO scene-availability query settings."""

    enabled: bool = False
    provider: str = "cdse"
    targets: tuple[str, ...] = ("regions", "kml_clusters")
    collections: tuple[str, ...] = ("sentinel-2", "sentinel-1")
    collection_names: dict = field(default_factory=lambda: {
        "sentinel-2": "SENTINEL-2",
        "sentinel-1": "SENTINEL-1",
        "sentinel-3-olci": "SENTINEL-3",
    })
    reference_date: str = ""
    temporal_buffer_days: int = 1
    spatial_buffer_km: float = 50.0
    max_cloud_cover_pct: float = 30.0
    result_limit: int = 50
    batch_result_limit: int = 10
    batch_log_interval: int = 10
    top_n_per_target: Optional[int] = None
    time_diff_decimal_places: int = 2
    provider_cfg: EOProviderConfig = field(
        default_factory=EOProviderConfig,
    )
    geometry: EOGeometryConfig = field(default_factory=EOGeometryConfig)
    csv_columns: EOCSVColumns = field(default_factory=EOCSVColumns)
    retry: RetrySettings = field(default_factory=RetrySettings)

    @classmethod
    def from_dict(cls, raw: dict) -> "EOConfig":
        """Flatten the nested provider/geometry/csv/retry YAML blocks.

        Args:
            raw (dict): Raw ``eo:`` mapping.

        Returns:
            EOConfig: Populated configuration.
        """
        flat = {
            key: val for key, val in raw.items()
            if key not in ("cdse", "geometry", "csv_columns", "retry")
        }
        flat["provider_cfg"] = _from_mapping(
            EOProviderConfig, raw.get("cdse", {}),
        )
        flat["geometry"] = _from_mapping(
            EOGeometryConfig, raw.get("geometry", {}),
        )
        flat["csv_columns"] = _from_mapping(
            EOCSVColumns, raw.get("csv_columns", {}),
        )
        flat["retry"] = _from_mapping(RetrySettings, raw.get("retry", {}))
        return _from_mapping(cls, flat)


@dataclass(frozen=True)
class WebknossosConfig:
    """WEBKNOSSOS export, local-serve, and remote-upload settings."""

    enabled: bool = False
    dataset_name: str = "hexcover"
    dataset_dir: str = "output/webknossos"
    layers: tuple[str, ...] = ("regions", "overlaps", "kml_clusters")
    voxel_deg: float = 0.02
    voxel_size_nm: tuple[float, ...] = (1000.0, 1000.0, 1000.0)
    padding_deg: float = 0.5
    max_raster_pixels: int = 8_000_000
    downsample: bool = True
    heat_scale: int = 255
    binary_data_dir: str = ""
    upload: bool = False
    url: str = "http://localhost:9000"
    token_env: str = "WK_TOKEN"

    @classmethod
    def from_dict(cls, raw: dict) -> "WebknossosConfig":
        """Flatten the nested ``local:`` / ``remote:`` YAML blocks.

        Args:
            raw (dict): Raw ``webknossos:`` mapping.

        Returns:
            WebknossosConfig: Populated configuration.
        """
        local = raw.get("local", {})
        remote = raw.get("remote", {})
        flat = {
            key: val for key, val in raw.items()
            if key not in ("local", "remote")
        }
        flat["binary_data_dir"] = local.get("binary_data_dir", "")
        flat["upload"] = remote.get("upload", False)
        flat["url"] = remote.get("url", "http://localhost:9000")
        flat["token_env"] = remote.get("token_env", "WK_TOKEN")
        return _from_mapping(cls, flat)


@dataclass(frozen=True)
class PrecisionConfig:
    """Decimal-place precision for reported metrics."""

    area_decimal_places: int = 6
    distance_decimal_places: int = 6


@dataclass(frozen=True)
class OutputPaths:
    """Filesystem destinations for pipeline outputs."""

    regions_geojson: str = "output/region_hexagons.geojson"
    combined_geojson: str = "output/combined_hexagons.geojson"
    overlaps_geojson: str = "output/overlap_hexagons.geojson"
    bias_report: str = "output/geo_bias_report.json"
    kml_hulls_geojson: str = "output/kml_hulls.geojson"
    kml_hexagons_geojson: str = "output/kml_cluster_hexagons.geojson"
    eo_scenes_csv: str = "output/eo_scenes.csv"
    eo_footprints_geojson: str = "output/eo_scene_footprints.geojson"


_SECTIONS: dict[str, type] = {
    "pipeline": PipelineSettings,
    "overlap": OverlapSettings,
    "limits": LimitsConfig,
    "overpass": OverpassConfig,
    "hull": HullConfig,
    "gjk": GJKConfig,
    "bias": BiasConfig,
    "cluster": ClusterConfig,
    "hierarchy": HierarchyConfig,
    "precision": PrecisionConfig,
    "output": OutputPaths,
}


@dataclass(frozen=True)
class Config:
    """Aggregate configuration for every pipeline subsystem."""

    pipeline: PipelineSettings
    regions: tuple[RegionConfig, ...]
    overlap: OverlapSettings
    limits: LimitsConfig
    overpass: OverpassConfig
    hull: HullConfig
    gjk: GJKConfig
    bias: BiasConfig
    cluster: ClusterConfig
    hierarchy: HierarchyConfig
    precision: PrecisionConfig
    output: OutputPaths
    kml: KMLConfig
    eo: EOConfig = field(default_factory=EOConfig)
    webknossos: WebknossosConfig = field(
        default_factory=WebknossosConfig,
    )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load the full configuration from a YAML file.

        Args:
            path (str): Path to config.yaml.

        Returns:
            Config: Populated configuration.

        Raises:
            ValueError: If region entries are missing or duplicated.
        """
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        assert isinstance(raw, dict), "config root must be a mapping"

        sections = {
            key: _from_mapping(section_cls, raw.get(key, {}))
            for key, section_cls in _SECTIONS.items()
        }
        regions = cls._build_regions(raw.get("regions", []))
        kml_raw = raw.get("kml", {})
        source_raw = dict(kml_raw.get("source", {}))
        retry_raw = source_raw.pop("retry", {})
        source = _from_mapping(
            KMLSourceConfig,
            {**source_raw, "retry": _from_mapping(RetrySettings, retry_raw)},
        )
        kml = KMLConfig(enabled=kml_raw.get("enabled", True), source=source)
        eo = EOConfig.from_dict(raw.get("eo", {}))
        wk = WebknossosConfig.from_dict(raw.get("webknossos", {}))
        return cls(
            regions=regions, kml=kml, eo=eo, webknossos=wk, **sections,
        )

    @staticmethod
    def _build_regions(raw: list) -> tuple[RegionConfig, ...]:
        """Build and validate the region list from YAML.

        Args:
            raw (list): Raw ``regions:`` entries.

        Returns:
            tuple[RegionConfig, ...]: Validated regions.

        Raises:
            ValueError: On missing names/IDs or duplicate names.
        """
        assert isinstance(raw, list), "regions must be a list"
        regions = tuple(_from_mapping(RegionConfig, entry) for entry in raw)
        names = [region.name for region in regions]
        if len(names) != len(set(names)):
            raise ValueError("region names must be unique")
        for region in regions:
            if not region.name or region.relation_id <= 0:
                raise ValueError(
                    f"invalid region entry: {region.name!r}"
                    f" / {region.relation_id}"
                )
        return regions