import logging
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from quant_platform.ingestion.base import BaseFetcher
from quant_platform.config import SP500_WIKIPEDIA_URL, DEFAULT_TICKERS

logger = logging.getLogger(__name__)

class YFinanceFetcher(BaseFetcher):
    """
    Fetcher implementation using the yfinance library.
    Fetches daily pricing and volume data from Yahoo Finance.
    """

    def fetch_sp500_tickers(self) -> list[str]:
        """
        Dynamically fetches the current list of S&P 500 tickers from Wikipedia.
        Falls back to a default hardcoded list if the request fails.
        """
        logger.info("Fetching S&P 500 tickers from Wikipedia...")
        try:
            tables = pd.read_html(SP500_WIKIPEDIA_URL)
            sp500_table = tables[0]
            tickers = sp500_table["Symbol"].tolist()
            
            # Clean tickers (Wikipedia uses '.' for class shares, Yahoo Finance uses '-')
            tickers = [t.replace(".", "-") for t in tickers]
            
            logger.info(f"Successfully retrieved {len(tickers)} S&P 500 tickers from Wikipedia.")
            return tickers
        except Exception as e:
            logger.warning(
                f"Failed to fetch S&P 500 tickers from Wikipedia ({e}). "
                f"Falling back to default test tickers."
            )
            return DEFAULT_TICKERS

    def fetch_daily_data(
        self, 
        tickers: list[str], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetches daily historical pricing and volume data for a list of tickers.
        """
        logger.info(f"Starting yfinance data download for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        all_data = []
        
        for symbol in tqdm(tickers, desc="Downloading Market Data"):
            try:
                # Fetch data using yfinance download
                # We download symbol-by-symbol to ensure granular error handling and avoid whole-batch failures
                df = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date, 
                    progress=False
                )
                
                if df.empty:
                    logger.debug(f"No data returned for ticker {symbol}")
                    continue
                
                # Reset index to bring Date into columns
                df = df.reset_index()
                
                # Flatten MultiIndex columns or tuples if present
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                df.columns = [str(col).lower() for col in df.columns]
                
                # If index was reset and called 'index' or similar, rename to 'date'
                if "index" in df.columns and "date" not in df.columns:
                    df = df.rename(columns={"index": "date"})
                
                # yfinance 0.2.x download format could return MultiIndex or simple columns.
                # If we fetch single symbol, it is a standard DataFrame.
                required_cols = ["date", "open", "high", "low", "close", "volume"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    logger.warning(f"Ticker {symbol} DataFrame missing columns {missing}, skipping.")
                    continue
                
                # Handle Adjusted Close column
                # If 'adj close' or 'adj_close' exists, use it. Otherwise, use close.
                adj_close_col = None
                for col in df.columns:
                    if "adj close" in col or "adj_close" in col:
                        adj_close_col = col
                        break
                
                # Build the cleaned dataframe
                cleaned_df = pd.DataFrame()
                cleaned_df["date"] = pd.to_datetime(df["date"]).dt.date
                cleaned_df["symbol"] = symbol
                cleaned_df["open"] = df["open"].astype(float)
                cleaned_df["high"] = df["high"].astype(float)
                cleaned_df["low"] = df["low"].astype(float)
                cleaned_df["close"] = df["close"].astype(float)
                
                if adj_close_col:
                    cleaned_df["adj_close"] = df[adj_close_col].astype(float)
                else:
                    cleaned_df["adj_close"] = df["close"].astype(float)
                    
                cleaned_df["volume"] = df["volume"].astype(int)
                
                # Drop rows with any NaN values in critical columns
                cleaned_df = cleaned_df.dropna(subset=["open", "high", "low", "close", "volume"])
                
                all_data.append(cleaned_df)
                
            except Exception as e:
                logger.error(f"Error fetching data for ticker {symbol}: {e}")
                continue
                
        if not all_data:
            logger.warning("No data was successfully fetched for any ticker.")
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
            
        # Concatenate all results
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Download complete. Fetched {len(final_df)} daily price records.")
        return final_df
