"""data_access: adjusted-price loading, filters, and matrix alignment."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from quant_platform.data_access import load_aligned_matrices


def _seed(db_path, prices, signals):
    """Insert explicit price/signal rows so each test controls both tables."""
    conn = duckdb.connect(str(db_path))
    # A fresh `signals` table carries only key columns; the feature column is
    # normally added by the factory's migration, so do the same here.
    existing = {row[1] for row in conn.execute("PRAGMA table_info('signals')").fetchall()}
    if "rsi" not in existing:
        conn.execute('ALTER TABLE signals ADD COLUMN "rsi" DOUBLE')
    conn.register("prices", prices)
    conn.execute("""
        INSERT INTO daily_prices (symbol, date, open, high, low, close, adj_close, volume)
        SELECT symbol, date, open, high, low, close, adj_close, volume FROM prices
    """)
    conn.register("sigs", signals)
    conn.execute("""
        INSERT INTO signals (symbol, date, rsi)
        SELECT symbol, date, rsi FROM sigs
    """)
    conn.unregister("prices")
    conn.unregister("sigs")
    conn.close()


def _price_rows(symbol, dates, closes, adj):
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.to_datetime(dates).date,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "adj_close": adj,
            "volume": [1000] * len(dates),
        }
    )


def _signal_rows(symbol, dates, rsi_values):
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.to_datetime(dates).date,
            "rsi": rsi_values,
        }
    )


DATES = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


@pytest.fixture()
def split_db(empty_db):
    """One symbol whose adjusted close deliberately differs from raw close."""
    prices = pd.concat(
        [
            _price_rows("AAA", DATES, [10.0, 10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 5.0, 5.0, 5.0]),
            _price_rows("BBB", DATES, [20.0] * 5, [20.0] * 5),
        ],
        ignore_index=True,
    )
    signals = pd.concat(
        [
            _signal_rows("AAA", DATES, [50.0] * 5),
            _signal_rows("BBB", DATES, [50.0] * 5),
        ],
        ignore_index=True,
    )
    _seed(empty_db, prices, signals)
    return empty_db


def test_close_matrix_uses_adjusted_prices(split_db):
    # AAA has a 2:1 split on Jan 3: raw close stays 10 while adj_close halves.
    close, _ = load_aligned_matrices(["rsi"])
    assert close["AAA"].tolist() == [10.0, 10.0, 5.0, 5.0, 5.0]


def test_filters_push_down_to_sql(split_db):
    close, matrices = load_aligned_matrices(["rsi"], start="2024-01-03", end="2024-01-04")
    assert list(close.index.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-04"]
    assert matrices["rsi"].shape == close.shape

    close_b, _ = load_aligned_matrices(["rsi"], symbols=["BBB"])
    assert list(close_b.columns) == ["BBB"]


def test_misaligned_frames_intersect_on_dates_and_symbols(empty_db):
    # BBB has prices but no signals, and CCC has signals but no prices:
    # only AAA survives in the aligned output.
    prices = _price_rows("AAA", DATES, [10.0] * 5, [10.0] * 5)
    prices = pd.concat([prices, _price_rows("BBB", DATES, [20.0] * 5, [20.0] * 5)], ignore_index=True)
    signals = _signal_rows("AAA", DATES[1:], [50.0] * 4)  # starts one day later
    signals = pd.concat([signals, _signal_rows("CCC", DATES[1:], [50.0] * 4)], ignore_index=True)
    _seed(empty_db, prices, signals)

    close, matrices = load_aligned_matrices(["rsi"])
    assert list(close.columns) == ["AAA"]
    assert list(close.index.strftime("%Y-%m-%d")) == DATES[1:]
    assert matrices["rsi"].shape == close.shape


def test_unknown_indicator_raises_actionable_error(split_db):
    with pytest.raises(ValueError, match="not found in 'signals'"):
        load_aligned_matrices(["rsi", "nope"])


def test_unsafe_indicator_name_is_rejected(split_db):
    with pytest.raises(ValueError, match="Unsafe SQL"):
        load_aligned_matrices(["rsi; DROP TABLE signals; --"])
