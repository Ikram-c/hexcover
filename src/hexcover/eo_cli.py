"""Standalone CLI for single-point and batch EO availability queries."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import Config
from .eo_client import build_eo_client
from .eo_query import (
    batch_query_from_csv,
    query_availability,
    strip_private_fields,
)
from .utils import AtomicWriter


def main() -> None:
    """Parse CLI arguments and dispatch to ``query`` or ``batch``."""
    parser = argparse.ArgumentParser(
        description="hexcover-eo - satellite scene availability queries",
    )
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser(
        "query", help="Query availability for a single point",
    )
    p_query.add_argument("--lat", type=float, required=True)
    p_query.add_argument("--lon", type=float, required=True)
    p_query.add_argument(
        "--date", type=str, required=True,
        help="ISO datetime e.g. 2026-07-01T12:00",
    )
    p_query.add_argument("--collections", nargs="+", default=None)
    p_query.add_argument("--output", type=Path, default=None)

    p_batch = sub.add_parser("batch", help="Batch query from CSV")
    p_batch.add_argument("csv", type=Path)
    p_batch.add_argument("--collections", nargs="+", default=None)
    p_batch.add_argument(
        "--output", type=Path, default=Path("eo_scenes.csv"),
    )

    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    client = build_eo_client(cfg.eo)

    if args.command == "query":
        records = query_availability(
            lat=args.lat, lon=args.lon,
            target_date=datetime.fromisoformat(args.date),
            client=client, cfg=cfg.eo, collections=args.collections,
        )
    else:
        records = batch_query_from_csv(
            str(args.csv), client, cfg.eo, collections=args.collections,
        )

    records = strip_private_fields(records)
    print(f"Found {len(records)} scenes")
    for record in records[:10]:
        print(
            f"  {record['collection']:12s}"
            f"  {record['scene_datetime']}"
            f"  dt={record['time_diff_hours']}h"
            f"  cloud={record['cloud_cover']}"
        )
    if args.command == "batch" or args.output:
        out = args.output or Path("eo_scenes.csv")
        AtomicWriter.write_csv(records, str(out))


if __name__ == "__main__":
    main()