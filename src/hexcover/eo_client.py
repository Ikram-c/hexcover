"""Registry-driven Earth-observation catalogue clients (CDSE first)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator, Optional

import requests

from .config import EOConfig
from .decorators import register
from .utils import retry_with_backoff

EO_PROVIDERS: dict[str, Callable[..., "EOClient"]] = {}


@dataclass(frozen=True)
class SatelliteScene:
    """One scene returned from a provider catalogue query.

    ``acquired`` replaces the original ``datetime`` field name, which
    shadowed the stdlib module.
    """

    id: str
    collection: str
    acquired: datetime
    cloud_cover: Optional[float]
    geometry: dict
    download_url: str
    thumbnail_url: Optional[str] = None


class EOClient:
    """Base catalogue client; providers register concrete subclasses."""

    def iter_scenes(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Iterator[SatelliteScene]:
        """Yield scenes matching the filter."""
        raise NotImplementedError


@register(EO_PROVIDERS, "cdse")
class CDSEClient(EOClient):
    """Copernicus Data Space OData client with bounded retries.

    Credentials are read from the environment variables named in the
    provider config; when absent the client runs unauthenticated,
    which is sufficient for catalogue queries.
    """

    def __init__(
        self,
        config: EOConfig,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Prepare the session and authenticate when credentials exist.

        Args:
            config (EOConfig): EO query settings.
            session (Optional[requests.Session]): Injectable session.
        """
        self._cfg = config
        self._provider = config.provider_cfg
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": self._provider.user_agent},
        )
        self.authenticated = False
        username = os.environ.get(self._provider.username_env, "")
        password = os.environ.get(self._provider.password_env, "")
        if username and password:
            self._authenticate(username, password)

    def _authenticate(self, username: str, password: str) -> None:
        """Exchange credentials for a bearer token, tolerating failure."""
        try:
            resp = self._session.post(
                self._provider.auth_url,
                data={
                    "client_id": self._provider.client_id,
                    "username": username,
                    "password": password,
                    "grant_type": "password",
                },
                timeout=self._provider.request_timeout_s,
            )
            resp.raise_for_status()
            token = resp.json()["access_token"]
            self._session.headers.update(
                {"Authorization": f"Bearer {token}"},
            )
            self.authenticated = True
            print("  CDSE authentication successful")
        except requests.RequestException as exc:
            print(f"  CDSE authentication failed: {exc}")

    def iter_scenes(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Iterator[SatelliteScene]:
        """Yield scenes for one collection; yields nothing on failure.

        Args:
            collection (str): Collection key (e.g. ``sentinel-2``).
            bbox (tuple[float, float, float, float]): (W, S, E, N) bbox.
            start_date (datetime): Window start (naive UTC).
            end_date (datetime): Window end (naive UTC).
            max_cloud_cover (Optional[float]): Cloud cap override (pct).
            limit (Optional[int]): Page-size override.

        Yields:
            SatelliteScene: One parsed scene per catalogue entry.
        """
        assert start_date <= end_date, "start_date must precede end_date"
        cloud_cap = (
            max_cloud_cover if max_cloud_cover is not None
            else self._cfg.max_cloud_cover_pct
        )
        page_size = limit or self._cfg.result_limit
        assert page_size > 0, "page size must be positive"
        name = self._cfg.collection_names.get(collection, collection)
        url = self._build_query_url(
            collection, name, bbox, start_date, end_date,
            cloud_cap, page_size,
        )
        try:
            payload = retry_with_backoff(
                lambda: self._fetch(url),
                self._cfg.retry,
                f"CDSE {collection}",
            )
        except requests.RequestException as exc:
            print(f"    CDSE query failed for {collection}: {exc}")
            return
        for item in payload.get("value", []):
            yield self._parse_scene(item, name)

    def _fetch(self, url: str) -> dict:
        """Execute one GET and return the parsed JSON payload."""
        resp = self._session.get(
            url, timeout=self._provider.request_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def _build_query_url(
        self,
        collection_key: str,
        collection_name: str,
        bbox: tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        cloud_cap: float,
        page_size: int,
    ) -> str:
        """Compose an OData ``$filter`` URL for the query parameters."""
        min_lon, min_lat, max_lon, max_lat = bbox
        polygon = (
            f"{min_lon} {min_lat},{max_lon} {min_lat},"
            f"{max_lon} {max_lat},{min_lon} {max_lat},"
            f"{min_lon} {min_lat}"
        )
        filters = [
            f"Collection/Name eq '{collection_name}'",
            f"ContentDate/Start ge {start_date.isoformat()}Z",
            f"ContentDate/Start le {end_date.isoformat()}Z",
            (
                f"OData.CSC.Intersects(area=geography'SRID=4326;"
                f"POLYGON(({polygon}))')"
            ),
        ]
        if "sentinel-2" in collection_key.lower():
            filters.append(
                f"Attributes/OData.CSC.DoubleAttribute/any("
                f"att:att/Name eq 'cloudCover' and"
                f" att/OData.CSC.DoubleAttribute/Value le {cloud_cap})"
            )
        base = self._provider.base_url
        return (
            f"{base}/Products?$filter=" + " and ".join(filters)
            + f"&$top={page_size}&$orderby=ContentDate/Start desc"
        )

    def _parse_scene(
        self, item: dict, collection_name: str,
    ) -> SatelliteScene:
        """Convert one OData ``Products`` entry into a scene record."""
        cloud = next(
            (
                attr.get("Value") for attr in item.get("Attributes", [])
                if attr.get("Name") == "cloudCover"
            ),
            None,
        )
        assets = item.get("Assets", [])
        return SatelliteScene(
            id=item["Id"],
            collection=collection_name,
            acquired=datetime.fromisoformat(
                item["ContentDate"]["Start"].replace("Z", "")
            ),
            cloud_cover=cloud,
            geometry=item.get("GeoFootprint", {}),
            download_url=(
                f"{self._provider.base_url}"
                f"/Products({item['Id']})/$value"
            ),
            thumbnail_url=(
                assets[0].get("DownloadLink") if assets else None
            ),
        )


def build_eo_client(
    config: EOConfig,
    session: Optional[requests.Session] = None,
) -> EOClient:
    """Construct an EO client via the provider registry.

    Args:
        config (EOConfig): EO settings, including ``provider``.
        session (Optional[requests.Session]): Injectable session.

    Returns:
        EOClient: Constructed client.

    Raises:
        ValueError: If the provider is not registered.
    """
    builder = EO_PROVIDERS.get(config.provider.lower())
    if builder is None:
        raise ValueError(
            f"unknown eo.provider {config.provider!r};"
            f" available: {sorted(EO_PROVIDERS)}"
        )
    return builder(config, session=session)