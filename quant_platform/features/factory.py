import importlib
import inspect
import logging
import pkgutil
import re
from pathlib import Path

import pandas as pd

from quant_platform.db.connection import get_db_connection
from quant_platform.db.schema import init_db
from quant_platform.features.base import BaseSignal

logger = logging.getLogger(__name__)

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Columns managed by the schema itself, never written by a signal module.
RESERVED_COLUMNS = {"symbol", "date", "calculated_at", "ingested_at"}


class SignalFactory:
    """
    SignalFactory orchestrates the feature engineering pipeline.
    It:
    1. Dynamically discovers and instantiates all subclasses of BaseSignal in the features package.
    2. Fetches daily price data from DuckDB.
    3. Groups prices by symbol and computes features for each.
    4. Merges computed features and upserts them into the signals table in DuckDB.
    5. Performs automatic schema migrations (ALTER TABLE) if custom new signals add new columns.
    """

    def __init__(self):
        self._signals: dict[str, BaseSignal] = {}
        self.discover_and_register_signals()

    @property
    def signals(self) -> dict[str, BaseSignal]:
        """Read-only view of the registered signal generators."""
        return dict(self._signals)

    def register(self, signal: BaseSignal) -> None:
        """
        Manually registers a signal generator.
        """
        self._signals[signal.name] = signal
        logger.info("Registered signal generator: %s", signal.name)

    def discover_and_register_signals(self) -> None:
        """
        Dynamically imports all modules under features/ and auto-registers
        any subclass of BaseSignal defined in that module.
        """
        logger.info("Starting dynamic discovery of alpha signals...")
        package_path = Path(__file__).resolve().parent

        for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
            if module_name in {"base", "factory"}:
                continue

            full_module_name = f"quant_platform.features.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
            except Exception as module_err:
                logger.error("Error importing module %s: %s", full_module_name, module_err)
                continue

            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls is BaseSignal or not issubclass(cls, BaseSignal):
                    continue
                # Only register classes *defined* in this module, so an imported
                # base class is not registered twice.
                if cls.__module__ != full_module_name:
                    continue
                try:
                    self.register(cls())
                except Exception as inst_err:
                    logger.error(
                        "Could not instantiate %s from %s: %s", cls.__name__, module_name, inst_err
                    )

    def compute_all(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all registered signals/features.
        """
        if price_df.empty:
            logger.warning("Empty price dataframe passed to feature computation.")
            return pd.DataFrame()

        all_symbol_features = []

        for symbol, group in price_df.groupby("symbol"):
            symbol_df = group.sort_values("date").copy()

            symbol_features_df = pd.DataFrame({
                "symbol": symbol_df["symbol"],
                "date": symbol_df["date"],
            })

            for signal_name, signal in self._signals.items():
                try:
                    missing_cols = [
                        c for c in signal.required_columns if c not in symbol_df.columns
                    ]
                    if missing_cols:
                        logger.warning(
                            "Ticker %s missing columns %s required by %s. Skipping signal.",
                            symbol,
                            missing_cols,
                            signal_name,
                        )
                        continue

                    computed_signal = signal.compute(symbol_df)
                    computed_signal = self._validate_signal_output(
                        signal_name, symbol, computed_signal, len(symbol_df)
                    )

                    # Suffix duplicate columns instead of silently renaming them
                    # to _x/_y, which would hide two signals fighting for a name.
                    overlap = [
                        c for c in computed_signal.columns
                        if c in symbol_features_df.columns and c not in {"symbol", "date"}
                    ]
                    if overlap:
                        raise ValueError(
                            f"columns {overlap} already produced by another signal"
                        )

                    symbol_features_df = pd.merge(
                        symbol_features_df,
                        computed_signal,
                        on=["symbol", "date"],
                        how="left",
                        validate="one_to_one",
                    )
                except Exception as signal_err:
                    logger.error(
                        "Error computing signal '%s' for symbol '%s': %s",
                        signal_name,
                        symbol,
                        signal_err,
                    )
                    continue

            all_symbol_features.append(symbol_features_df)

        if not all_symbol_features:
            return pd.DataFrame()

        return pd.concat(all_symbol_features, ignore_index=True)

    @staticmethod
    def _validate_signal_output(
        signal_name: str, symbol: str, computed: pd.DataFrame, expected_rows: int
    ) -> pd.DataFrame:
        """Reject malformed signal output before it reaches the database."""
        if not isinstance(computed, pd.DataFrame):
            raise TypeError(f"{signal_name} returned {type(computed).__name__}, expected DataFrame")

        missing_keys = [k for k in ("symbol", "date") if k not in computed.columns]
        if missing_keys:
            raise ValueError(f"{signal_name} output missing key columns {missing_keys}")

        if len(computed) != expected_rows:
            raise ValueError(
                f"{signal_name} returned {len(computed)} rows for {symbol}, expected {expected_rows}"
            )

        return computed

    @staticmethod
    def _feature_columns(features_df: pd.DataFrame) -> list[str]:
        """Feature columns to persist, excluding keys and reserved metadata."""
        return [
            c for c in features_df.columns
            if c not in RESERVED_COLUMNS
        ]

    @staticmethod
    def _quote_identifier(name: str) -> str:
        """Validate a column name so it can be safely interpolated into DDL."""
        if not IDENTIFIER_RE.match(name):
            raise ValueError(f"Unsafe column name produced by a signal: {name!r}")
        return f'"{name}"'

    def run_pipeline(self) -> int:
        """
        Executes the entire feature engineering pipeline:
        1. Reads prices from DuckDB.
        2. Computes features.
        3. Migrates DB columns if necessary.
        4. Upserts to signals table.
        """
        logger.info("Executing feature engineering pipeline...")

        with get_db_connection(read_only=True) as conn:
            table_check = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'daily_prices'"
            ).fetchone()

            if not table_check:
                logger.error("Table 'daily_prices' does not exist. Run ingestion pipeline first.")
                return 0

            price_df = conn.execute("SELECT * FROM daily_prices ORDER BY symbol, date").df()

        if price_df.empty:
            logger.warning("No data found in 'daily_prices' table. Run ingestion pipeline first.")
            return 0

        features_df = self.compute_all(price_df)
        if features_df.empty:
            logger.warning("No features were computed.")
            return 0

        feature_cols = self._feature_columns(features_df)
        if not feature_cols:
            logger.warning("Signals produced no feature columns; nothing to persist.")
            return 0

        logger.info("Computed %d feature rows. Saving to DuckDB...", len(features_df))

        try:
            with get_db_connection(read_only=False) as conn:
                init_db(conn)
                self._migrate_columns(conn, feature_cols)
                written = self._upsert_features(conn, features_df, feature_cols)

                total_rows = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]

            logger.info(
                "Successfully computed and saved features. "
                "Rows written: %d. Total rows in 'signals': %d",
                written,
                total_rows,
            )
            return written

        except Exception as e:
            logger.error("Failed to persist computed features to DuckDB: %s", e)
            raise

    def _migrate_columns(self, conn, feature_cols: list[str]) -> None:
        """Add any signal column missing from the signals table."""
        db_col_names = [row[1] for row in conn.execute("PRAGMA table_info('signals')").fetchall()]

        for col in feature_cols:
            if col not in db_col_names:
                logger.info("Database Migration: adding column '%s' to 'signals'.", col)
                conn.execute(
                    f"ALTER TABLE signals ADD COLUMN {self._quote_identifier(col)} DOUBLE"
                )

    def _upsert_features(self, conn, features_df: pd.DataFrame, feature_cols: list[str]) -> int:
        """Insert new keys and update existing ones.

        ``INSERT OR REPLACE`` deletes and re-inserts the whole row, so a signal
        that failed for one symbol would blank out that symbol's previously good
        columns. An explicit UPDATE-then-INSERT only touches the columns we
        actually computed.
        """
        conn.register("features_df", features_df)
        try:
            key_cols = ["symbol", "date"]

            conn.execute("BEGIN TRANSACTION")

            update_cols = [
                c for c in feature_cols
                if c not in key_cols
            ]
            if update_cols:
                assignments = ", ".join(
                    f"{self._quote_identifier(c)} = src.{self._quote_identifier(c)}"
                    for c in update_cols
                )
                conn.execute(f"""
                    UPDATE signals AS dst
                    SET {assignments}, calculated_at = CURRENT_TIMESTAMP
                    FROM features_df AS src
                    WHERE dst.symbol = src.symbol AND dst.date = src.date
                """)

            all_cols = key_cols + update_cols
            cols_str = ", ".join(self._quote_identifier(c) for c in all_cols)
            select_str = ", ".join(f"src.{self._quote_identifier(c)}" for c in all_cols)

            conn.execute(f"""
                INSERT INTO signals ({cols_str})
                SELECT {select_str}
                FROM features_df AS src
                WHERE NOT EXISTS (
                    SELECT 1 FROM signals AS dst
                    WHERE dst.symbol = src.symbol AND dst.date = src.date
                )
            """)

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.unregister("features_df")

        return len(features_df)
