import logging

import duckdb

logger = logging.getLogger(__name__)


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Initializes the DuckDB database schema.

    ``signals`` carries only its key columns here: every feature column is
    added by the signal factory's schema migration, so adding a signal module
    never requires touching this DDL.
    """
    logger.info("Initializing database schema...")

    # 1. Create daily_prices table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE NOT NULL,
            adj_close DOUBLE,
            volume BIGINT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        )
    """)
    logger.info("Ensured table 'daily_prices' exists.")

    # 2. Create signals table; feature columns arrive via migration.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        )
    """)
    logger.info("Ensured table 'signals' exists.")
