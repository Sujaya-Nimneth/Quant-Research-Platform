"""Shared fixtures: isolated DuckDB databases and synthetic price series."""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_platform.db.schema import init_db  # noqa: E402


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    """Point the platform's DB_PATH at a throwaway DuckDB file.

    Every module reads the path via ``config.DB_PATH`` at call time, so a
    single patch isolates the whole platform.
    """
    path = tmp_path / "test.db"
    monkeypatch.setattr("quant_platform.config.DB_PATH", path)
    return path


@pytest.fixture()
def empty_db(db_path):
    conn = duckdb.connect(str(db_path))
    init_db(conn)
    conn.close()
    return db_path


@pytest.fixture()
def price_frame() -> pd.DataFrame:
    """Two synthetic symbols with a deterministic trend plus oscillation."""
    rows = []
    start = dt.date(2024, 1, 1)
    for symbol, offset in (("AAA", 0.0), ("BBB", 1.5)):
        price = 100.0
        for i in range(120):
            # Deterministic pseudo-random walk.
            price *= 1.0 + 0.004 * math.sin(i / 3.0) + 0.001 * math.cos(i / 7.0)
            day = start + dt.timedelta(days=i)
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "open": price * 0.995,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price + offset,
                    "adj_close": price + offset,
                    "volume": 1_000_000 + i,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def populated_db(empty_db, price_frame):
    conn = duckdb.connect(str(empty_db))
    conn.register("prices", price_frame)
    conn.execute("""
        INSERT INTO daily_prices (symbol, date, open, high, low, close, adj_close, volume)
        SELECT symbol, date, open, high, low, close, adj_close, volume FROM prices
    """)
    conn.unregister("prices")
    conn.close()
    return empty_db
