"""Registry-driven sources of (lat, lon) points for the KML pipeline."""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import numpy as np
import requests

from .config import KMLSourceConfig, RetrySettings
from .decorators import register
from .utils import retry_with_backoff

KML_SOURCE_BUILDERS: dict[str, Callable[..., "KMLPointSource"]] = {}

KML_NAMESPACES = (
    "http://earth.google.com/kml/2.1",
    "http://www.opengis.net/kml/2.2",
)
KMZ_MAGIC = b"PK\x03\x04"

_NGFF_TRANSFORMS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "scale": lambda pts, vec: pts * vec,
    "translation": lambda pts, vec: pts + vec,
}


def _iter_coords_from_root(root: ET.Element) -> Iterator[tuple[float, float]]:
    """Yield (lat, lon) for every Placemark/Point in a KML tree."""
    for namespace in KML_NAMESPACES:
        for placemark in root.iter(f"{{{namespace}}}Placemark"):
            elem = placemark.find(
                f"{{{namespace}}}Point/{{{namespace}}}coordinates"
            )
            if elem is None or not elem.text:
                continue
            parts = elem.text.strip().split(",")
            if len(parts) >= 2:
                yield (float(parts[1]), float(parts[0]))


def _parse_kml_bytes(content: bytes) -> ET.Element:
    """Parse a KML or KMZ payload into a root XML element.

    Args:
        content (bytes): Raw KML/KMZ bytes.

    Returns:
        ET.Element: Root XML element.

    Raises:
        ValueError: If a KMZ archive contains no .kml entries.
    """
    assert len(content) > 0, "empty KML payload"
    if content.startswith(KMZ_MAGIC):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [n for n in archive.namelist()
                     if n.lower().endswith(".kml")]
            if not names:
                raise ValueError("KMZ archive contains no .kml files")
            kml_name = "doc.kml" if "doc.kml" in names else names[0]
            with archive.open(kml_name) as handle:
                return ET.parse(handle).getroot()
    return ET.fromstring(content)


class KMLPointSource:
    """Base source of (lat, lon) points for the KML pipeline."""

    def iter_points(self) -> Iterator[tuple[float, float]]:
        """Yield (lat, lon) pairs lazily."""
        raise NotImplementedError

    def parse_points(self) -> np.ndarray:
        """Materialise ``iter_points`` into an (N, 2) float64 array."""
        return np.fromiter(
            self.iter_points(), dtype=np.dtype((np.float64, 2)),
        )


class FileKMLSource(KMLPointSource):
    """Read points from a local KML or KMZ file."""

    def __init__(self, path: Union[str, Path]) -> None:
        """Store the path; existence is checked at iteration time."""
        self._path = Path(path)

    def iter_points(self) -> Iterator[tuple[float, float]]:
        """Stream coordinates from the parsed KML/KMZ tree.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"KML source not found: {self._path}")
        root = _parse_kml_bytes(self._path.read_bytes())
        yield from _iter_coords_from_root(root)


class URLKMLSource(KMLPointSource):
    """Fetch a KML/KMZ document over HTTP with bounded retries."""

    def __init__(
        self,
        url: str,
        timeout_s: int,
        headers: dict,
        retry: RetrySettings,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Store request parameters and the retry policy."""
        assert url, "URL source requires a URL"
        self._url = url
        self._timeout_s = timeout_s
        self._headers = headers
        self._retry = retry
        self._session = session

    def iter_points(self) -> Iterator[tuple[float, float]]:
        """GET with retries, parse, and yield coordinates."""
        content = retry_with_backoff(
            self._fetch, self._retry, f"GET {self._url}",
        )
        yield from _iter_coords_from_root(_parse_kml_bytes(content))

    def _fetch(self) -> bytes:
        """Execute one HTTP GET and return the raw body."""
        session = self._session or requests.Session()
        resp = session.get(
            self._url, timeout=self._timeout_s, headers=self._headers,
        )
        resp.raise_for_status()
        return resp.content


class InMemoryKMLSource(KMLPointSource):
    """Wrap an already-materialised (N, 2) (lat, lon) array."""

    def __init__(
        self, points: Union[np.ndarray, Iterable[tuple[float, float]]],
    ) -> None:
        """Coerce and validate the stored array.

        Raises:
            ValueError: If the input is not (N, 2).
        """
        coerced = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        if coerced.ndim != 2 or coerced.shape[1] != 2:
            raise ValueError("InMemoryKMLSource expects an (N, 2) array")
        self._points = coerced

    def iter_points(self) -> Iterator[tuple[float, float]]:
        """Yield each row as a (lat, lon) tuple."""
        for row in self._points:
            yield (float(row[0]), float(row[1]))

    def parse_points(self) -> np.ndarray:
        """Return a defensive copy of the stored array."""
        return self._points.copy()


class CallableKMLSource(KMLPointSource):
    """Wrap any zero-argument callable yielding (lat, lon) pairs."""

    def __init__(
        self, producer: Callable[[], Iterable[tuple[float, float]]],
    ) -> None:
        """Store the producer callable."""
        self._producer = producer

    def iter_points(self) -> Iterator[tuple[float, float]]:
        """Invoke the producer and yield its results."""
        yield from self._producer()


class ZarrKMLSource(KMLPointSource):
    """Read points from a Zarr store as one (N, 2) array or two 1D arrays."""

    def __init__(self, cfg: KMLSourceConfig) -> None:
        """Store connection parameters; the store is opened lazily."""
        assert cfg.store, "Zarr source requires a store"
        self._cfg = cfg

    def parse_points(self) -> np.ndarray:
        """Open the store with retries and load the configured arrays."""
        return retry_with_backoff(
            self._load, self._cfg.retry, f"Zarr read {self._cfg.store}",
        )

    def iter_points(self) -> Iterator[tuple[float, float]]:
        """Yield points from a one-shot bulk read."""
        for row in self.parse_points():
            yield (float(row[0]), float(row[1]))

    def _load(self) -> np.ndarray:
        """Open the group and read points."""
        return self._read_arrays(self._open_group())

    def _open_group(self) -> Any:
        """Open the Zarr group at the configured store and sub-path."""
        import zarr
        kwargs: dict[str, Any] = {"mode": "r"}
        if self._cfg.storage_options:
            kwargs["storage_options"] = dict(self._cfg.storage_options)
        group = zarr.open_group(self._cfg.store, **kwargs)
        if self._cfg.group_path:
            group = group[self._cfg.group_path]
        return group

    def _read_arrays(self, group: Any) -> np.ndarray:
        """Materialise the configured point arrays as (N, 2) float64.

        Raises:
            ValueError: On shape mismatches.
        """
        if self._cfg.points_array:
            arr = np.asarray(
                group[self._cfg.points_array][:], dtype=np.float64,
            )
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    f"expected (N, 2) at '{self._cfg.points_array}',"
                    f" got {arr.shape}"
                )
            return arr
        lats = np.asarray(group[self._cfg.lat_array][:], dtype=np.float64)
        lons = np.asarray(group[self._cfg.lon_array][:], dtype=np.float64)
        if lats.shape != lons.shape:
            raise ValueError(
                f"lat/lon shape mismatch: {lats.shape} vs {lons.shape}"
            )
        return np.column_stack([lats.ravel(), lons.ravel()])


class NGFFZarrKMLSource(ZarrKMLSource):
    """ZarrKMLSource that applies OME-NGFF coordinate transformations."""

    def _load(self) -> np.ndarray:
        """Load points and apply declared scale/translation transforms."""
        group = self._open_group()
        points = self._read_arrays(group)
        if not self._cfg.apply_transforms:
            return points
        return self._apply(points, self._read_transforms(group))

    def _read_transforms(self, group: Any) -> list[dict]:
        """Pull ``coordinateTransformations`` for the configured scale."""
        multiscales = group.attrs.get("multiscales")
        if not multiscales:
            return []
        try:
            datasets = multiscales[0].get("datasets", [])
            if self._cfg.scale_index >= len(datasets):
                return []
            return datasets[self._cfg.scale_index].get(
                "coordinateTransformations", []
            )
        except (KeyError, IndexError, TypeError, AttributeError):
            return []

    @staticmethod
    def _apply(points: np.ndarray, transforms: list[dict]) -> np.ndarray:
        """Apply transforms in order via the NGFF dispatch table."""
        result = points.astype(np.float64, copy=True)
        for transform in transforms:
            ttype = transform.get("type", "")
            handler = _NGFF_TRANSFORMS.get(ttype)
            if handler is None:
                continue
            vec = np.asarray(transform.get(ttype, []), dtype=np.float64)
            if len(vec) == result.shape[1]:
                result = handler(result, vec)
        return result


@register(KML_SOURCE_BUILDERS, "file")
def _build_file_source(cfg: KMLSourceConfig) -> FileKMLSource:
    """Build a file-backed source from config.

    Raises:
        ValueError: If no path is configured.
    """
    if not cfg.path:
        raise ValueError("kml.source.type='file' requires kml.source.path")
    return FileKMLSource(cfg.path)


@register(KML_SOURCE_BUILDERS, "url")
def _build_url_source(cfg: KMLSourceConfig) -> URLKMLSource:
    """Build an HTTP-backed source from config.

    Raises:
        ValueError: If no URL is configured.
    """
    if not cfg.url:
        raise ValueError("kml.source.type='url' requires kml.source.url")
    return URLKMLSource(
        url=cfg.url,
        timeout_s=cfg.timeout_s,
        headers=dict(cfg.headers),
        retry=cfg.retry,
    )


@register(KML_SOURCE_BUILDERS, "zarr")
def _build_zarr_source(cfg: KMLSourceConfig) -> ZarrKMLSource:
    """Build a Zarr-backed source from config."""
    return ZarrKMLSource(cfg)


@register(KML_SOURCE_BUILDERS, "ngff_zarr")
def _build_ngff_source(cfg: KMLSourceConfig) -> NGFFZarrKMLSource:
    """Build an NGFF-Zarr-backed source from config."""
    return NGFFZarrKMLSource(cfg)


def build_kml_source(cfg: KMLSourceConfig) -> KMLPointSource:
    """Construct a KML source via the registry.

    Args:
        cfg (KMLSourceConfig): YAML-derived source config.

    Returns:
        KMLPointSource: Constructed source.

    Raises:
        ValueError: If the type is not registered.
    """
    builder = KML_SOURCE_BUILDERS.get(cfg.type.lower())
    if builder is None:
        raise ValueError(
            f"unknown kml.source.type {cfg.type!r};"
            f" available: {sorted(KML_SOURCE_BUILDERS)}"
        )
    return builder(cfg)