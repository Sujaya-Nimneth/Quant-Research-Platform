import logging
import duckdb
from contextlib import contextmanager
from typing import Generator
from quant_platform.config import DB_PATH

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection(read_only: bool = False) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Context manager to safely open and close a connection to the local DuckDB database.
    Ensures file locks are freed appropriately.
    """
    conn = None
    try:
        # Open connection
        conn = duckdb.connect(database=str(DB_PATH), read_only=read_only)
        yield conn
    except Exception as e:
        logger.error(f"Error accessing DuckDB at {DB_PATH}: {e}")
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing DuckDB connection: {e}")
