import logging
from collections.abc import Generator
from contextlib import contextmanager

import duckdb

from quant_platform import config

logger = logging.getLogger(__name__)


def _connect(read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB file, with actionable errors for the common failures.

    ``config.DB_PATH`` is read at call time (not import time) so tests and
    alternative deployments can redirect it by patching one attribute.
    """
    if not read_only:
        # The data directory may not exist on a first run.
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        return duckdb.connect(database=str(config.DB_PATH), read_only=read_only)
    except duckdb.Error as e:
        if "lock" in str(e).lower() or "conflict" in str(e).lower():
            raise RuntimeError(
                f"Cannot open {config.DB_PATH}: another process holds the database "
                "(DuckDB allows a single writer). Close the dashboard or the other "
                "pipeline stage and retry."
            ) from e
        raise


@contextmanager
def get_db_connection(read_only: bool = False) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Context manager to safely open and close a connection to the local DuckDB database.
    Ensures file locks are freed appropriately.
    """
    conn = _connect(read_only)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            logger.exception("Error closing DuckDB connection.")
