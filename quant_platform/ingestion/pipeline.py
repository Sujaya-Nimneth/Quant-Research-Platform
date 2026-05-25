import logging
import pandas as pd
from quant_platform.db.connection import get_db_connection
from quant_platform.db.schema import init_db
from quant_platform.ingestion.base import BaseFetcher

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the data ingestion pipeline:
    1. Initializes database tables if they do not exist.
    2. Uses a fetcher to download pricing data.
    3. Bulk upserts the data into DuckDB.
    """

    def __init__(self, fetcher: BaseFetcher):
        self.fetcher = fetcher

    def run(
        self, 
        tickers: list[str] = None, 
        start_date: str = "2024-01-01", 
        end_date: str = None
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

        Returns:
        --------
        int
            Number of rows ingested into the database.
        """
        # Set default end date if not provided
        if end_date is None:
            import datetime
            end_date = datetime.date.today().strftime("%Y-%m-%d")

        # Get tickers if not provided
        if tickers is None:
            # If the fetcher has a specific method to fetch S&P 500 list, use it
            if hasattr(self.fetcher, "fetch_sp500_tickers"):
                tickers = self.fetcher.fetch_sp500_tickers()
            else:
                from quant_platform.config import DEFAULT_TICKERS
                tickers = DEFAULT_TICKERS

        # Initialize the database schema
        with get_db_connection(read_only=False) as conn:
            init_db(conn)

        # Fetch market data
        df = self.fetcher.fetch_daily_data(
            tickers=tickers, 
            start_date=start_date, 
            end_date=end_date
        )

        if df.empty:
            logger.warning("No data was fetched. Database remains unchanged.")
            return 0

        # Upsert the data into DuckDB
        # DuckDB handles Pandas dataframes directly from Python local scope by variable name reference
        logger.info(f"Ingesting {len(df)} records into DuckDB...")
        try:
            with get_db_connection(read_only=False) as conn:
                # We register the dataframe so DuckDB can query it
                # DuckDB's local query engine resolves the variable `df` in the scope
                conn.execute("""
                    INSERT OR REPLACE INTO daily_prices (
                        symbol, date, open, high, low, close, adj_close, volume
                    )
                    SELECT 
                        symbol, date, open, high, low, close, adj_close, volume 
                    FROM df
                """)
                
                # Fetch row count
                result = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()
                total_rows = result[0] if result else 0
                
            logger.info(f"Successfully upserted data. Total rows in 'daily_prices' table: {total_rows}")
            return len(df)
            
        except Exception as e:
            logger.error(f"Failed to ingest dataframe into DuckDB: {e}")
            raise
