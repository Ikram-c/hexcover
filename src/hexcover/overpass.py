"""Resilient Overpass client with mirror rotation and rate limiting."""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import requests

from .config import OverpassConfig


class OverpassClient:
    """Resilient Overpass client with mirror rotation and rate limiting."""

    def __init__(self, config: OverpassConfig) -> None:
        """Store config and prepare static request headers."""
        assert len(config.urls) > 0, "at least one Overpass URL required"
        self._cfg = config
        self._url_index = 0
        self._last_request: Optional[float] = None
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": config.user_agent,
            "Referer": config.referer,
        }

    def fetch_relation_nodes(self, relation_id: int) -> np.ndarray:
        """Fetch all outer-way nodes of an OSM relation.

        Args:
            relation_id (int): OSM relation ID.

        Returns:
            np.ndarray: (N, 2) (lat, lon) array.
        """
        assert relation_id > 0, "relation_id must be positive"
        query = (
            f"[out:json][timeout:{self._cfg.timeout_s}];"
            f"relation({relation_id});"
            'way(r:"outer");node(w);out body;'
        )
        data = self._request(query)
        coords = np.fromiter(
            (
                (el["lat"], el["lon"])
                for el in data["elements"]
                if el["type"] == "node"
            ),
            dtype=np.dtype((np.float64, 2)),
        )
        print(f"    fetched {len(coords)} nodes")
        return coords

    def _request(self, query: str) -> dict:
        """POST the query with bounded retries and mirror rotation.

        Args:
            query (str): Overpass QL query.

        Returns:
            dict: Parsed JSON payload.

        Raises:
            Exception: The final failure once retries are exhausted.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self._cfg.max_retries + 1):
            url = self._cfg.urls[self._url_index % len(self._cfg.urls)]
            self._throttle()
            print(
                f"  querying Overpass (attempt {attempt + 1}"
                f"/{self._cfg.max_retries + 1})"
            )
            try:
                self._last_request = time.monotonic()
                resp = requests.post(
                    url,
                    data={"data": query},
                    headers=self._headers,
                    timeout=self._cfg.timeout_s + self._cfg.timeout_buffer_s,
                )
                if resp.status_code in self._cfg.retryable_status_codes:
                    last_error = requests.HTTPError(
                        f"{resp.status_code} {resp.reason}", response=resp,
                    )
                    self._back_off(attempt, resp.status_code)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                self._back_off(attempt, type(exc).__name__)
        assert last_error is not None, "retry loop exited without result"
        raise last_error

    def _throttle(self) -> None:
        """Sleep until the configured rate-limit interval has elapsed."""
        if self._last_request is None:
            return
        elapsed = time.monotonic() - self._last_request
        remaining = self._cfg.rate_limit_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _back_off(self, attempt: int, reason: Union[int, str]) -> None:
        """Rotate mirrors and sleep before the next attempt, if any remain."""
        if attempt >= self._cfg.max_retries:
            print(f"    exhausted retries ({reason})")
            return
        self._url_index += 1
        wait = self._cfg.backoff_base_s * (
            self._cfg.backoff_multiplier ** attempt
        )
        print(f"    {reason}: backing off {wait:.0f}s")
        time.sleep(wait)