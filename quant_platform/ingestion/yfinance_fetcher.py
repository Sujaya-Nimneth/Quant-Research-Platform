import logging
import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from quant_platform.config import DEFAULT_TICKERS, SP500_WIKIPEDIA_URL
from quant_platform.ingestion.base import BaseFetcher

logger = logging.getLogger(__name__)

# Yahoo Finance and Wikipedia reject requests without a browser-like agent.
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; quant-platform/0.2)"}

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0


class TickerListError(RuntimeError):
    """Raised when the S&P 500 constituent list cannot be retrieved."""


class YFinanceFetcher(BaseFetcher):
    """
    Fetcher implementation using the yfinance library.
    Fetches daily pricing and volume data from Yahoo Finance.

    Prices are requested with ``auto_adjust=False`` so ``adj_close`` is a real
    split/dividend-adjusted series rather than a duplicate of ``close``.
    """

    def __init__(
        self,
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        request_pause_seconds: float = 0.0,
    ):
        self.retries = max(1, retries)
        self.backoff_seconds = backoff_seconds
        self.request_pause_seconds = request_pause_seconds

    def fetch_sp500_tickers(self, strict: bool = False) -> list[str]:
        """
        Dynamically fetches the current list of S&P 500 tickers from Wikipedia.

        Parameters
        ----------
        strict:
            When True, raise :class:`TickerListError` instead of silently
            falling back to the small default list. Scheduled/full runs should
            use this so a failed scrape cannot masquerade as a full ingest.
        """
        logger.info("Fetching S&P 500 tickers from Wikipedia...")
        try:
            tables = pd.read_html(SP500_WIKIPEDIA_URL, storage_options=HTTP_HEADERS)
            sp500_table = tables[0]
            tickers = sp500_table["Symbol"].tolist()

            # Wikipedia uses '.' for class shares; Yahoo Finance uses '-'.
            tickers = [str(t).replace(".", "-").strip() for t in tickers]
            tickers = [t for t in tickers if t]

            if len(tickers) < 400:
                raise TickerListError(
                    f"Wikipedia returned only {len(tickers)} tickers; expected ~500."
                )

            logger.info("Successfully retrieved %d S&P 500 tickers from Wikipedia.", len(tickers))
            return tickers
        except Exception as e:
            message = (
                f"Failed to fetch S&P 500 tickers from Wikipedia ({e}). "
                f"Falling back to the {len(DEFAULT_TICKERS)}-ticker default list."
            )
            if strict:
                raise TickerListError(message) from e
            logger.warning(message)
            return list(DEFAULT_TICKERS)

    def fetch_daily_data(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetches daily historical pricing and volume data for a list of tickers.
        """
        logger.info(
            "Starting yfinance download for %d tickers from %s to %s...",
            len(tickers),
            start_date,
            end_date,
        )

        all_data = []
        failures: list[str] = []

        for symbol in tqdm(tickers, desc="Downloading Market Data"):
            try:
                cleaned = self._fetch_symbol(symbol, start_date, end_date)
            except Exception as e:
                logger.error("Error fetching data for ticker %s: %s", symbol, e)
                failures.append(symbol)
                continue

            if cleaned is None or cleaned.empty:
                logger.debug("No data returned for ticker %s", symbol)
                continue

            all_data.append(cleaned)

            if self.request_pause_seconds:
                time.sleep(self.request_pause_seconds)

        if failures:
            logger.warning(
                "Failed to fetch %d/%d tickers: %s",
                len(failures),
                len(tickers),
                ", ".join(failures[:20]) + (" ..." if len(failures) > 20 else ""),
            )

        if not all_data:
            logger.warning("No data was successfully fetched for any ticker.")
            return pd.DataFrame(
                columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
            )

        final_df = pd.concat(all_data, ignore_index=True)
        logger.info("Download complete. Fetched %d daily price records.", len(final_df))
        return final_df

    def _fetch_symbol(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Download and normalise one symbol, retrying transient failures."""
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                    threads=False,
                )
                return self._normalize(symbol, df)
            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    sleep_for = self.backoff_seconds * (2 ** (attempt - 1))
                    logger.debug(
                        "Retrying %s after error (attempt %d/%d) in %.1fs: %s",
                        symbol,
                        attempt,
                        self.retries,
                        sleep_for,
                        e,
                    )
                    time.sleep(sleep_for)

        raise last_error if last_error else RuntimeError(f"Unknown failure for {symbol}")

    @staticmethod
    def _normalize(symbol: str, df: pd.DataFrame) -> pd.DataFrame | None:
        """Convert a raw yfinance frame into the platform's canonical shape."""
        if df is None or df.empty:
            return None

        df = df.reset_index()

        # yfinance may return a MultiIndex (field, ticker) even for one symbol.
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [str(col).lower() for col in df.columns]

        if "index" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"index": "date"})
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        required_cols = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning("Ticker %s frame missing columns %s, skipping.", symbol, missing)
            return None

        adj_close_col = next(
            (c for c in df.columns if "adj close" in c or "adj_close" in c), None
        )

        cleaned = pd.DataFrame()
        cleaned["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        cleaned["symbol"] = symbol
        for column in ("open", "high", "low", "close"):
            cleaned[column] = pd.to_numeric(df[column], errors="coerce")

        if adj_close_col:
            cleaned["adj_close"] = pd.to_numeric(df[adj_close_col], errors="coerce")
        else:
            cleaned["adj_close"] = cleaned["close"]

        # Volume can legitimately be NaN/zero on halted sessions; keep it
        # nullable instead of raising on an int cast.
        cleaned["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

        cleaned = cleaned.dropna(subset=["date", "open", "high", "low", "close"])
        cleaned = cleaned[cleaned["close"] > 0]
        return cleaned
