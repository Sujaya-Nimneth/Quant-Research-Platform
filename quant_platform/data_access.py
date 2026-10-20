"""Shared helpers for loading aligned price/indicator matrices from DuckDB.

Both the backtest engine and the Streamlit dashboard need the exact same
pivoted, index-aligned matrices. Keeping that logic in one place guarantees
they cannot drift apart.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_platform.db.connection import get_db_connection

logger = logging.getLogger(__name__)

PRICE_TABLE = "daily_prices"
SIGNAL_TABLE = "signals"


def table_exists(conn, table_name: str) -> bool:
    """Return True when ``table_name`` exists in the main schema."""
    row = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [table_name],
    ).fetchone()
    return row is not None


def require_tables() -> None:
    """Raise a helpful error if the ingestion/feature pipelines have not run."""
    with get_db_connection(read_only=True) as conn:
        missing = [t for t in (PRICE_TABLE, SIGNAL_TABLE) if not table_exists(conn, t)]
    if missing:
        raise ValueError(
            f"Missing table(s): {', '.join(missing)}. "
            "Run `python -m quant_platform.main ingest --test` followed by "
            "`python -m quant_platform.main features` first."
        )


def load_aligned_matrices(indicators: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load close prices plus the requested indicator columns as wide matrices.

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
    require_tables()

    with get_db_connection(read_only=True) as conn:
        price_df = conn.execute(
            f"SELECT symbol, date, close FROM {PRICE_TABLE} ORDER BY date, symbol"
        ).df()

        selected = ", ".join(["symbol", "date", *indicators])
        signal_df = conn.execute(
            f"SELECT {selected} FROM {SIGNAL_TABLE} ORDER BY date, symbol"
        ).df()

    if price_df.empty or signal_df.empty:
        raise ValueError(
            "No rows found in 'daily_prices' or 'signals'. Run the ingestion and "
            "feature pipelines first."
        )

    price_df["date"] = pd.to_datetime(price_df["date"])
    signal_df["date"] = pd.to_datetime(signal_df["date"])

    close = price_df.pivot(index="date", columns="symbol", values="close")
    matrices: dict[str, pd.DataFrame] = {}
    for name in indicators:
        matrices[name] = signal_df.pivot(index="date", columns="symbol", values=name)

    # Intersect dates and symbols across every frame so arithmetic is well defined.
    for matrix in matrices.values():
        close, matrix = close.align(matrix, join="inner", axis=0)
        close, matrix = close.align(matrix, join="inner", axis=1)

    close.index = pd.DatetimeIndex(close.index)
    close = close.sort_index()
    for name, matrix in matrices.items():
        matrix.index = pd.DatetimeIndex(matrix.index)
        matrices[name] = matrix.reindex(index=close.index, columns=close.columns)

    logger.info(
        "Loaded aligned data for %d symbols across %d sessions.",
        len(close.columns),
        len(close),
    )
    return close, matrices
