"""Bounded retries, atomic writes, and iterative union-find."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import requests

from .config import RetrySettings


def retry_with_backoff(
    operation: Callable[[], Any],
    settings: RetrySettings,
    description: str,
) -> Any:
    """Run ``operation`` with a bounded exponential-backoff retry loop.

    Args:
        operation (Callable[[], Any]): Zero-argument callable.
        settings (RetrySettings): Retry policy.
        description (str): Label used in log lines.

    Returns:
        Any: The operation's return value.

    Raises:
        Exception: The final failure once retries are exhausted.
    """
    assert settings.max_retries >= 0, "max_retries must be non-negative"
    last_error: Optional[Exception] = None
    for attempt in range(settings.max_retries + 1):
        try:
            return operation()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response else None
            if code not in settings.retryable_status_codes:
                raise
            last_error = exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
        if attempt < settings.max_retries:
            wait = settings.backoff_base_s * (
                settings.backoff_multiplier ** attempt
            )
            print(f"    {description}: retrying in {wait:.0f}s")
            time.sleep(wait)
    assert last_error is not None, "retry loop exited without a result"
    raise last_error


class AtomicWriter:
    """Atomic JSON/GeoJSON/CSV writer using a temp-file-and-rename strategy."""

    @staticmethod
    def write_json(payload: dict, path: str) -> int:
        """Write ``payload`` atomically and return the byte count.

        Args:
            payload (dict): JSON-serialisable payload.
            path (str): Destination path.

        Returns:
            int: Number of bytes written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=2)
        with tempfile.NamedTemporaryFile(
            "w", dir=target.parent, suffix=".tmp",
            delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, target)
        written = target.stat().st_size
        assert written > 0, f"empty write to {target}"
        print(f"  wrote {target} ({written} bytes)")
        return written

    @classmethod
    def write_features(cls, features: list[dict], path: str) -> int:
        """Write a GeoJSON FeatureCollection atomically.

        Args:
            features (list[dict]): GeoJSON features.
            path (str): Destination path.

        Returns:
            int: Number of bytes written.
        """
        payload = {"type": "FeatureCollection", "features": features}
        return cls.write_json(payload, path)

    @staticmethod
    def write_csv(records: list[dict], path: str) -> int:
        """Write dict records as CSV atomically, preserving key order.

        Args:
            records (list[dict]): Rows; fieldnames are the union of keys
                in first-appearance order.
            path (str): Destination path.

        Returns:
            int: Number of rows written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for record in records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)
        with tempfile.NamedTemporaryFile(
            "w", dir=target.parent, suffix=".tmp",
            delete=False, encoding="utf-8", newline="",
        ) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, target)
        print(f"  wrote {target} ({len(records)} rows)")
        return len(records)


class UnionFind:
    """Iterative union-find with path halving (no recursion)."""

    def __init__(self, size: int) -> None:
        """Initialise ``size`` singleton components.

        Args:
            size (int): Number of elements.
        """
        assert size >= 0, "size must be non-negative"
        self._parent = np.arange(size, dtype=np.int64)

    def find(self, node: int) -> int:
        """Return the component root of ``node``."""
        parent = self._parent
        assert 0 <= node < len(parent), "node out of range"
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(self, a: int, b: int) -> None:
        """Merge the components containing ``a`` and ``b``."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b

    def groups(self) -> dict[int, list[int]]:
        """Return ``root -> member indices`` for every component."""
        out: dict[int, list[int]] = {}
        for idx in range(len(self._parent)):
            out.setdefault(self.find(idx), []).append(idx)
        return out