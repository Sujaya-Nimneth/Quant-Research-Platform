import datetime
import logging

import pandas as pd

from quant_platform import config
from quant_platform.db.connection import get_db_connection
from quant_platform.db.schema import init_db
from quant_platform.ingestion.base import BaseFetcher

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]


class IngestionPipeline:
    """
    Orchestrates the data ingestion pipeline:
    1. Initializes database tables if they do not exist.
    2. Uses a fetcher to download pricing data.
    3. Bulk upserts the data into DuckDB.

    Runs are incremental by default: each symbol is fetched from the day after
    its own latest stored date, so a daily refresh downloads one session per
    symbol and a lagging symbol (halted stock, transient fetch failure) is
    backfilled instead of permanently stranded.
    """

    def __init__(self, fetcher: BaseFetcher, batch_size: int = 500_000):
        self.fetcher = fetcher
        self.batch_size = batch_size

    def run(
        self,
        tickers: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
        strict_tickers: bool = False,
    ) -> int:
        """
        Runs the ingestion pipeline.

        Parameters:
        -----------
        tickers : list[str], optional
            Tickers to fetch. If None, the fetcher is asked for S&P 500 tickers.
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. Defaults to
            ``config.DEFAULT_START_DATE``.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (exclusive, matching yfinance).
            Defaults to tomorrow so a same-day run picks up today's session.
        full_refresh : bool
            Re-download the entire history instead of only new sessions.
        strict_tickers : bool
            Fail instead of silently substituting the small default ticker list.

        Returns:
        --------
        int
            Number of rows ingested into the database.
        """
        if start_date is None:
            start_date = config.DEFAULT_START_DATE
        if end_date is None:
            # yfinance's `end` is exclusive; defaulting to tomorrow means a
            # same-day run still ingests today's session.
            end_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        config.ensure_dirs()
        with get_db_connection(read_only=False) as conn:
            init_db(conn)

        if tickers is None:
            tickers = self._resolve_tickers(strict_tickers)

        if not tickers:
            logger.warning("No tickers to ingest.")
            return 0

        fetch_groups = self._fetch_groups(tickers, start_date, full_refresh)

        frames = []
        for fetch_start, group in sorted(fetch_groups.items()):
            logger.info("Fetching %d ticker(s) from %s...", len(group), fetch_start)
            frame = self.fetcher.fetch_daily_data(
                tickers=group,
                start_date=fetch_start,
                end_date=end_date,
            )
            if not frame.empty:
                frames.append(frame)

        if not frames:
            logger.warning("No data was fetched. Database remains unchanged.")
            return 0

        df = pd.concat(frames, ignore_index=True)
        return self._upsert(df)

    def _fetch_groups(self, tickers: list[str], start_date: str, full_refresh: bool) -> dict[str, list[str]]:
        """Group tickers by the date their incremental fetch should start from.

        Each symbol resumes from its own latest stored bar (re-fetching that
        bar, since Yahoo occasionally revises it and the upsert is
        idempotent), so one lagging symbol cannot strand the rest.
        """
        floor = datetime.date.fromisoformat(start_date)
        if full_refresh:
            return {start_date: list(tickers)}

        high_water = self._latest_stored_dates()
        groups: dict[str, list[str]] = {}
        for ticker in tickers:
            latest = high_water.get(ticker)
            resume = max(floor, latest) if latest is not None else floor
            groups.setdefault(resume.isoformat(), []).append(ticker)
        return groups

    def _resolve_tickers(self, strict: bool) -> list[str]:
        try:
            return self.fetcher.fetch_sp500_tickers(strict=strict)
        except NotImplementedError:
            logger.info("Fetcher does not provide S&P 500 discovery; using default ticker list.")
            return list(config.DEFAULT_TICKERS)

    @staticmethod
    def _latest_stored_dates() -> dict[str, datetime.date]:
        with get_db_connection(read_only=True) as conn:
            rows = conn.execute("SELECT symbol, MAX(date) FROM daily_prices GROUP BY symbol").fetchall()
        return {symbol: latest for symbol, latest in rows if latest is not None}

    def _upsert(self, df: pd.DataFrame) -> int:
        """Insert or replace daily price rows in batched transactions."""
        missing = [c for c in PRICE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Fetcher output is missing required columns: {missing}")

        df = df[PRICE_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.drop_duplicates(subset=["symbol", "date"], keep="last")

        logger.info("Ingesting %d records into DuckDB...", len(df))

        with get_db_connection(read_only=False) as conn:
            total_written = 0
            for start in range(0, len(df), self.batch_size):
                batch = df.iloc[start : start + self.batch_size]
                conn.register("batch_df", batch)
                try:
                    conn.execute("BEGIN TRANSACTION")
                    conn.execute("""
                        INSERT OR REPLACE INTO daily_prices (
                            symbol, date, open, high, low, close, adj_close, volume, ingested_at
                        )
                        SELECT symbol, date, open, high, low, close, adj_close, volume,
                               CURRENT_TIMESTAMP
                        FROM batch_df
                    """)
                    conn.execute("COMMIT")
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        logger.exception("Rollback failed; original error follows.")
                    raise
                finally:
                    conn.unregister("batch_df")
                total_written += len(batch)

            total_rows = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]

        logger.info(
            "Successfully upserted data. Rows written: %d. Total rows in 'daily_prices': %d",
            total_written,
            total_rows,
        )
        return total_written
