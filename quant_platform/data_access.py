"""Shared helpers for loading aligned price/indicator matrices from DuckDB.

Both the backtest engine and the Streamlit dashboard need the exact same
pivoted, index-aligned matrices. Keeping that logic in one place guarantees
they cannot drift apart.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_platform import config
from quant_platform.db.connection import get_db_connection
from quant_platform.db.identifiers import quote_identifier

logger = logging.getLogger(__name__)

PRICE_TABLE = "daily_prices"
SIGNAL_TABLE = "signals"


def table_exists(conn, table_name: str) -> bool:
    """Return True when ``table_name`` exists in the main schema."""
    row = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
        [table_name],
    ).fetchone()
    return row is not None


def _existing_columns(conn, table_name: str) -> set[str]:
    """Column names of ``table_name`` in the main schema."""
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = 'main' AND table_name = ?",
        [table_name],
    ).fetchall()
    return {row[0] for row in rows}


def _check_db_exists() -> None:
    """Raise a helpful error when the DuckDB file has never been created."""
    if not config.DB_PATH.exists():
        raise ValueError(
            f"Database not found at {config.DB_PATH}. "
            "Run `python -m quant_platform.main ingest --test` followed by "
            "`python -m quant_platform.main features` first."
        )


def _require_tables_on(conn) -> None:
    """Raise when either pipeline table is missing; expects the DB file to exist."""
    missing = [t for t in (PRICE_TABLE, SIGNAL_TABLE) if not table_exists(conn, t)]
    if missing:
        raise ValueError(
            f"Missing table(s): {', '.join(missing)}. "
            "Run `python -m quant_platform.main ingest --test` followed by "
            "`python -m quant_platform.main features` first."
        )


def require_tables() -> None:
    """Raise a helpful error if the ingestion/feature pipelines have not run."""
    _check_db_exists()
    with get_db_connection(read_only=True) as conn:
        _require_tables_on(conn)


def load_aligned_matrices(indicators: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load adjusted close prices plus the requested indicator columns as wide matrices.

    Prices are read from ``adj_close`` - the split/dividend-adjusted series -
    so a corporate action never looks like a price move to the signals, the
    backtest fills, or the dashboard benchmark. The raw ``close`` column stays
    in storage for reference only.

    Parameters
    ----------
    indicators:
        Column names from the ``signals`` table to pivot, e.g. ``["rsi"]``.

    Returns
    -------
    tuple
        ``(close, indicators)`` where ``close`` is a ``date x symbol`` frame and
        ``indicators`` maps each requested name to its own ``date x symbol``
        frame. All frames share identical index and columns.
    """
    _check_db_exists()
    quoted_indicators = [quote_identifier(name, kind="indicator column") for name in indicators]

    with get_db_connection(read_only=True) as conn:
        _require_tables_on(conn)
        available = _existing_columns(conn, SIGNAL_TABLE)
        missing = [name for name in indicators if name not in available]
        if missing:
            raise ValueError(
                f"Indicator column(s) {missing} not found in 'signals'. "
                "Re-run the feature pipeline (`python -m quant_platform.main features`) "
                "and check the strategy's required_indicators."
            )

        price_df = conn.execute(
            f"SELECT symbol, date, adj_close AS close FROM {PRICE_TABLE} ORDER BY date, symbol"
        ).df()
        selected = ", ".join(["symbol", "date", *quoted_indicators])
        signal_df = conn.execute(f"SELECT {selected} FROM {SIGNAL_TABLE} ORDER BY date, symbol").df()

    if price_df.empty or signal_df.empty:
        raise ValueError("No rows found in 'daily_prices' or 'signals'. Run the ingestion and feature pipelines first.")

    price_df["date"] = pd.to_datetime(price_df["date"])
    signal_df["date"] = pd.to_datetime(signal_df["date"])

    close = price_df.pivot(index="date", columns="symbol", values="close")
    matrices: dict[str, pd.DataFrame] = {}
    for name in indicators:
        matrices[name] = signal_df.pivot(index="date", columns="symbol", values=name)

    # Intersect dates and symbols across every frame so arithmetic is well defined.
    common_index = close.index
    common_columns = close.columns
    for matrix in matrices.values():
        common_index = common_index.intersection(matrix.index)
        common_columns = common_columns.intersection(matrix.columns)

    close = close.loc[common_index, common_columns].sort_index()
    for name, matrix in matrices.items():
        matrices[name] = matrix.reindex(index=close.index, columns=close.columns)

    logger.info(
        "Loaded aligned data for %d symbols across %d sessions.",
        len(close.columns),
        len(close),
    )
    return close, matrices
