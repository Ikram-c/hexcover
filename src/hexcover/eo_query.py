"""Single-point and batch scene-availability queries over an EOClient."""
from __future__ import annotations

import heapq
import itertools
import math
from datetime import datetime, timedelta
from typing import Iterable, Iterator, Optional

import pandas as pd

from .config import EOConfig, EOGeometryConfig
from .eo_client import EOClient, SatelliteScene


def compute_bbox(
    lat: float,
    lon: float,
    buffer_km: float,
    geom: EOGeometryConfig,
) -> tuple[float, float, float, float]:
    """Return a (W, S, E, N) bbox approximating ``buffer_km`` around a point.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        buffer_km (float): Half-width of the box in kilometres.
        geom (EOGeometryConfig): Conversion constants.

    Returns:
        tuple[float, float, float, float]: (min_lon, min_lat,
            max_lon, max_lat).
    """
    assert -90.0 <= lat <= 90.0, "latitude out of range"
    assert -180.0 <= lon <= 180.0, "longitude out of range"
    assert buffer_km > 0, "buffer must be positive"
    cos_lat = max(geom.min_cos_clamp, abs(math.cos(math.radians(lat))))
    lat_offset = buffer_km / geom.km_per_degree_lat
    lon_offset = buffer_km / (geom.km_per_degree_lat * cos_lat)
    return (
        lon - lon_offset, lat - lat_offset,
        lon + lon_offset, lat + lat_offset,
    )


def iter_scene_records(
    scenes: Iterable[SatelliteScene],
    target_date: datetime,
    collection: str,
    cfg: EOConfig,
    extra: Optional[dict] = None,
) -> Iterator[dict]:
    """Yield one result-row dict per scene with rounded time difference.

    Args:
        scenes (Iterable[SatelliteScene]): Scenes to convert.
        target_date (datetime): Reference timestamp.
        collection (str): Collection key for the output rows.
        cfg (EOConfig): Rounding and unit settings.
        extra (Optional[dict]): Base fields merged into every row.

    Yields:
        dict: One flat result row per scene.
    """
    base = extra or {}
    seconds_per_hour = cfg.geometry.seconds_per_hour
    for scene in scenes:
        diff = (
            (scene.acquired - target_date).total_seconds()
            / seconds_per_hour
        )
        yield {
            **base,
            "collection": collection,
            "scene_id": scene.id,
            "scene_datetime": scene.acquired.isoformat(),
            "time_diff_hours": round(
                abs(diff), cfg.time_diff_decimal_places,
            ),
            "cloud_cover": scene.cloud_cover,
            "download_url": scene.download_url,
            "_geometry": scene.geometry,
        }


def top_n_by_proximity(
    records: Iterator[dict], top_n: Optional[int],
) -> Iterator[dict]:
    """Yield only the N records with the smallest time difference.

    Args:
        records (Iterator[dict]): Candidate rows.
        top_n (Optional[int]): Cap; ``None`` passes everything through.

    Yields:
        dict: Selected rows.
    """
    if top_n is None:
        yield from records
        return
    assert top_n > 0, "top_n must be positive"
    yield from heapq.nsmallest(
        top_n, records, key=lambda record: record["time_diff_hours"],
    )


def query_availability(
    lat: float,
    lon: float,
    target_date: datetime,
    client: EOClient,
    cfg: EOConfig,
    collections: Optional[list[str]] = None,
    limit: Optional[int] = None,
    extra: Optional[dict] = None,
) -> list[dict]:
    """Query every collection around one point and return result rows.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        target_date (datetime): Reference timestamp (naive UTC).
        client (EOClient): Catalogue client.
        cfg (EOConfig): Query settings.
        collections (Optional[list[str]]): Override collection keys.
        limit (Optional[int]): Per-collection result cap override.
        extra (Optional[dict]): Base fields merged into every row.

    Returns:
        list[dict]: Result rows, top-N filtered when configured.
    """
    keys = list(collections or cfg.collections)
    assert len(keys) > 0, "at least one collection required"
    bbox = compute_bbox(lat, lon, cfg.spatial_buffer_km, cfg.geometry)
    start = target_date - timedelta(days=cfg.temporal_buffer_days)
    end = target_date + timedelta(days=cfg.temporal_buffer_days)

    streams = (
        iter_scene_records(
            client.iter_scenes(
                collection=key, bbox=bbox,
                start_date=start, end_date=end, limit=limit,
            ),
            target_date, key, cfg, extra=extra,
        )
        for key in keys
    )
    return list(top_n_by_proximity(
        itertools.chain.from_iterable(streams), cfg.top_n_per_target,
    ))


def scene_footprint_features(records: list[dict]) -> Iterator[dict]:
    """Yield GeoJSON features for records carrying a footprint geometry.

    Args:
        records (list[dict]): Rows from ``query_availability``.

    Yields:
        dict: One GeoJSON feature per footprinted scene.
    """
    for record in records:
        geometry = record.get("_geometry")
        if not geometry:
            continue
        props = {
            key: val for key, val in record.items()
            if key != "_geometry"
        }
        yield {
            "type": "Feature",
            "properties": {**props, "type": "eo_scene"},
            "geometry": geometry,
        }


def strip_private_fields(records: list[dict]) -> list[dict]:
    """Return records without underscore-prefixed internal fields."""
    return [
        {k: v for k, v in record.items() if not k.startswith("_")}
        for record in records
    ]


def batch_query_from_csv(
    csv_path: str,
    client: EOClient,
    cfg: EOConfig,
    collections: Optional[list[str]] = None,
) -> list[dict]:
    """Run ``query_availability`` for every valid row in a voyage CSV.

    Args:
        csv_path (str): Input CSV path.
        client (EOClient): Catalogue client.
        cfg (EOConfig): Query and column settings.
        collections (Optional[list[str]]): Override collection keys.

    Returns:
        list[dict]: Concatenated result rows across all voyages.
    """
    cols = cfg.csv_columns
    df = pd.read_csv(csv_path)
    df["start_datetime"] = pd.to_datetime(
        df[cols.date].astype(str) + " " + df[cols.time].astype(str),
        dayfirst=True, errors="coerce",
    )
    for col_name in (cols.latitude, cols.longitude):
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df = df.dropna(
        subset=["start_datetime", cols.latitude, cols.longitude],
    )
    print(f"  loaded {len(df)} records with valid coordinates")

    results: list[dict] = []
    for position, (idx, row) in enumerate(df.iterrows(), start=1):
        if position % cfg.batch_log_interval == 0:
            print(f"  processed {position}/{len(df)} records")
        base = {
            "voyage_id": row.get(cols.id, idx),
            "voyage_datetime": row["start_datetime"].isoformat(),
            "voyage_lat": float(row[cols.latitude]),
            "voyage_lon": float(row[cols.longitude]),
            "camera_id": row.get(cols.camera_id),
        }
        results.extend(query_availability(
            lat=float(row[cols.latitude]),
            lon=float(row[cols.longitude]),
            target_date=row["start_datetime"].to_pydatetime(),
            client=client, cfg=cfg, collections=collections,
            limit=cfg.batch_result_limit, extra=base,
        ))
    return results