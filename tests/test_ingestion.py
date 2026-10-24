"""Ingestion: normalization, incremental behaviour, and upsert semantics."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from quant_platform.ingestion.base import BaseFetcher
from quant_platform.ingestion.pipeline import IngestionPipeline
from quant_platform.ingestion.yfinance_fetcher import TickerListError, YFinanceFetcher


class StubFetcher(BaseFetcher):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[tuple[list[str], str, str]] = []

    def fetch_daily_data(self, tickers, start_date, end_date) -> pd.DataFrame:
        self.calls.append((list(tickers), start_date, end_date))
        if "symbol" not in self.frame.columns:
            return pd.DataFrame()
        wanted = set(tickers)
        return self.frame[self.frame["symbol"].isin(wanted)].copy()


class SplitByDateStubFetcher(BaseFetcher):
    """Serves different frames per symbol so each can lag by a different amount."""

    def __init__(self, frames_by_symbol: dict[str, pd.DataFrame]):
        self.frames = frames_by_symbol
        self.calls: list[tuple[list[str], str, str]] = []

    def fetch_daily_data(self, tickers, start_date, end_date) -> pd.DataFrame:
        self.calls.append((list(tickers), start_date, end_date))
        frames = [self.frames[symbol] for symbol in tickers if symbol in self.frames]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _frame(symbols=("AAA",), dates=("2024-01-02", "2024-01-03")) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        for i, day in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(day).date(),
                    "open": 10.0 + i,
                    "high": 11.0 + i,
                    "low": 9.0 + i,
                    "close": 10.5 + i,
                    "adj_close": 10.4 + i,
                    "volume": 1000 + i,
                }
            )
    return pd.DataFrame(rows)


def test_ingest_writes_rows(empty_db):
    pipeline = IngestionPipeline(StubFetcher(_frame()))
    written = pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01")

    assert written == 2
    conn = duckdb.connect(str(empty_db), read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0] == 2
    conn.close()


def test_second_run_is_incremental(empty_db):
    fetcher = StubFetcher(_frame())
    pipeline = IngestionPipeline(fetcher)
    pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01")
    pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01")

    _, first_start, _ = fetcher.calls[0]
    _, second_start, _ = fetcher.calls[1]
    assert first_start == "2024-01-01"
    assert second_start == "2024-01-03", "second run should resume from the latest stored bar"


def test_full_refresh_ignores_stored_dates(empty_db):
    fetcher = StubFetcher(_frame())
    pipeline = IngestionPipeline(fetcher)
    pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01")
    pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01", full_refresh=True)

    assert fetcher.calls[1][1] == "2024-01-01"


def test_lagging_symbol_is_fetched_from_own_high_water_mark(empty_db):
    # AAA is current through Jan 3 while BBB stalled at Jan 2. The incremental
    # run must resume BBB from its own bar; a global high-water mark would
    # strand BBB's Jan 3 session forever.
    fetcher = SplitByDateStubFetcher(
        {
            "AAA": _frame(symbols=("AAA",), dates=("2024-01-02", "2024-01-03")),
            "BBB": _frame(symbols=("BBB",), dates=("2024-01-02",)),
        }
    )
    pipeline = IngestionPipeline(fetcher)
    pipeline.run(tickers=["AAA", "BBB"], start_date="2024-01-01", end_date="2024-02-01")

    fetcher.frames = {
        "AAA": _frame(symbols=("AAA",), dates=("2024-01-02", "2024-01-03", "2024-01-04")),
        "BBB": _frame(symbols=("BBB",), dates=("2024-01-02", "2024-01-03", "2024-01-04")),
    }
    fetcher.calls.clear()
    pipeline.run(tickers=["AAA", "BBB"], start_date="2024-01-01", end_date="2024-02-01")

    starts = {tuple(symbols): start for symbols, start, _ in fetcher.calls}
    assert starts == {("AAA",): "2024-01-03", ("BBB",): "2024-01-02"}

    conn = duckdb.connect(str(empty_db), read_only=True)
    bbb_dates = [
        str(row[0]) for row in conn.execute("SELECT date FROM daily_prices WHERE symbol='BBB' ORDER BY date").fetchall()
    ]
    conn.close()
    assert bbb_dates == ["2024-01-02", "2024-01-03", "2024-01-04"]


def test_default_dates_cover_today(empty_db):
    # end_date defaults to tomorrow because yfinance's end bound is exclusive.
    import datetime as dt

    fetcher = StubFetcher(_frame())
    pipeline = IngestionPipeline(fetcher)
    pipeline.run(tickers=["AAA"])
    _, _, requested_end = fetcher.calls[0]
    expected = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    assert requested_end == expected


def test_upsert_updates_existing_rows_without_duplicating(empty_db):
    pipeline = IngestionPipeline(StubFetcher(_frame()))
    pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01")

    updated = _frame()
    updated.loc[0, "close"] = 999.0
    pipeline.fetcher = StubFetcher(updated)
    pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01")

    conn = duckdb.connect(str(empty_db), read_only=True)
    count = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    close = conn.execute("SELECT close FROM daily_prices WHERE symbol='AAA' AND date = DATE '2024-01-02'").fetchone()[0]
    conn.close()

    assert count == 2
    assert close == 999.0


def test_missing_columns_raise(empty_db):
    pipeline = IngestionPipeline(StubFetcher(pd.DataFrame({"symbol": ["AAA"]})))
    with pytest.raises(ValueError, match="missing required columns"):
        pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01")


def test_empty_fetch_is_a_no_op(empty_db):
    pipeline = IngestionPipeline(StubFetcher(pd.DataFrame()))
    assert pipeline.run(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01") == 0


def test_normalize_handles_nan_volume_and_positive_filter():
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "Open": [10.0, 10.0, 10.0],
            "High": [11.0, 11.0, 11.0],
            "Low": [9.0, 9.0, 9.0],
            "Close": [10.5, 0.0, 10.7],
            "Adj Close": [10.4, 0.0, 10.6],
            "Volume": [1000, None, 1200],
        }
    )
    result = YFinanceFetcher._normalize("AAA", raw)

    assert result is not None
    assert len(result) == 2, "non-positive closes are dropped"
    assert result["volume"].isna().sum() == 0
    assert str(result["volume"].dtype) == "Int64"


def test_normalize_returns_none_for_empty_frame():
    assert YFinanceFetcher._normalize("AAA", pd.DataFrame()) is None


def test_strict_ticker_discovery_raises_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(pd, "read_html", boom)
    with pytest.raises(TickerListError):
        YFinanceFetcher().fetch_sp500_tickers(strict=True)


def test_non_strict_ticker_discovery_falls_back(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(pd, "read_html", boom)
    tickers = YFinanceFetcher().fetch_sp500_tickers(strict=False)
    assert len(tickers) == 15


def test_ticker_discovery_rejects_suspiciously_short_list(monkeypatch):
    monkeypatch.setattr(pd, "read_html", lambda *a, **k: [pd.DataFrame({"Symbol": ["AAA", "BBB"]})])
    with pytest.raises(TickerListError):
        YFinanceFetcher().fetch_sp500_tickers(strict=True)
