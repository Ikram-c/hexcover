"""Registration and stage-logging decorators."""
from __future__ import annotations

import functools
import time
from typing import Any, Callable


def register(registry: dict[str, Any], name: str) -> Callable:
    """Register an object in a lookup table under ``name``.

    Args:
        registry (dict[str, Any]): Target registry.
        name (str): Registration key.

    Returns:
        Callable: Decorator that stores and returns the object unchanged.
    """
    def _wrap(obj: Any) -> Any:
        assert name not in registry, f"duplicate registration: {name}"
        registry[name] = obj
        return obj
    return _wrap


def stage(label: str) -> Callable:
    """Log a labelled, timed pipeline stage around a method call.

    Args:
        label (str): Human-readable stage name.

    Returns:
        Callable: Decorator wrapping the stage method.
    """
    def _wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _run(*args: Any, **kwargs: Any) -> Any:
            print(f"\n{label}")
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            print(f"  [{label}: {time.perf_counter() - start:.2f}s]")
            return result
        return _run
    return _wrap