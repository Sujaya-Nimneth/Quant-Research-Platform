import logging
import duckdb

logger = logging.getLogger(__name__)

def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Initializes the DuckDB database schema.
    Creates daily_prices and signals tables with composite primary keys.
    """
    logger.info("Initializing database schema...")
    
    # 1. Create daily_prices table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume BIGINT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        )
    """)
    logger.info("Table 'daily_prices' is ready.")

    # 2. Create signals table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            symbol VARCHAR,
            date DATE,
            macd_line DOUBLE,
            macd_signal DOUBLE,
            macd_hist DOUBLE,
            roc DOUBLE,
            momentum DOUBLE,
            rsi DOUBLE,
            bb_upper DOUBLE,
            bb_lower DOUBLE,
            bb_percent DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        )
    """)
    logger.info("Table 'signals' is ready.")
