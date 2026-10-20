import datetime
import logging

import pandas as pd

from quant_platform.config import DEFAULT_TICKERS
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
    its latest stored date, so a daily refresh downloads one session instead of
    the full history.
    """

    def __init__(self, fetcher: BaseFetcher, batch_size: int = 500_000):
        self.fetcher = fetcher
        self.batch_size = batch_size

    def run(
        self,
        tickers: list[str] = None,
        start_date: str = "2024-01-01",
        end_date: str = None,
        full_refresh: bool = False,
        strict_tickers: bool = False,
    ) -> int:
        """
        Runs the ingestion pipeline.

        Parameters:
        -----------
        tickers : list[str], optional
            Tickers to fetch. If None, the fetcher is asked for S&P 500 tickers.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. Defaults to current date.
        full_refresh : bool
            Re-download the entire history instead of only new sessions.
        strict_tickers : bool
            Fail instead of silently substituting the small default ticker list.

        Returns:
        --------
        int
            Number of rows ingested into the database.
        """
        if end_date is None:
            end_date = datetime.date.today().strftime("%Y-%m-%d")

        with get_db_connection(read_only=False) as conn:
            init_db(conn)

        if tickers is None:
            tickers = self._resolve_tickers(strict_tickers)

        if not tickers:
            logger.warning("No tickers to ingest.")
            return 0

        fetch_start = start_date
        if not full_refresh:
            latest = self._latest_stored_date()
            if latest is not None:
                # Re-fetch the last stored session too: Yahoo occasionally
                # revises the most recent bar, and the upsert is idempotent.
                fetch_start = max(
                    datetime.date.fromisoformat(start_date),
                    latest,
                ).isoformat()
                logger.info("Incremental ingestion from %s (latest stored bar).", fetch_start)

        df = self.fetcher.fetch_daily_data(
            tickers=tickers,
            start_date=fetch_start,
            end_date=end_date,
        )

        if df.empty:
            logger.warning("No data was fetched. Database remains unchanged.")
            return 0

        return self._upsert(df)

    def _resolve_tickers(self, strict: bool) -> list[str]:
        if hasattr(self.fetcher, "fetch_sp500_tickers"):
            try:
                return self.fetcher.fetch_sp500_tickers(strict=strict)
            except TypeError:
                # Fetcher implementations that predate the `strict` flag.
                return self.fetcher.fetch_sp500_tickers()
        return list(DEFAULT_TICKERS)

    @staticmethod
    def _latest_stored_date() -> datetime.date | None:
        with get_db_connection(read_only=True) as conn:
            row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
        return row[0] if row and row[0] is not None else None

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
                batch = df.iloc[start:start + self.batch_size]
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
                    conn.execute("ROLLBACK")
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
