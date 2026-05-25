import logging
import importlib
import pkgutil
import inspect
from pathlib import Path
import pandas as pd
from quant_platform.db.connection import get_db_connection
from quant_platform.db.schema import init_db
from quant_platform.features.base import BaseSignal

logger = logging.getLogger(__name__)

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

    def register(self, signal: BaseSignal) -> None:
        """
        Manually registers a signal generator.
        """
        self._signals[signal.name] = signal
        logger.info(f"Registered signal generator: {signal.name}")

    def discover_and_register_signals(self) -> None:
        """
        Dynamically imports all modules under features/ and auto-registers
        any subclass of BaseSignal.
        """
        logger.info("Starting dynamic discovery of alpha signals...")
        try:
            # We locate the package containing this file
            package_path = Path(__file__).resolve().parent
            
            # Walk and import all modules in this folder
            for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
                if module_name in ["base", "factory"]:
                    continue
                    
                full_module_name = f"quant_platform.features.{module_name}"
                try:
                    module = importlib.import_module(full_module_name)
                    # Find all classes that inherit from BaseSignal
                    for name, cls in inspect.getmembers(module, inspect.isclass):
                        if issubclass(cls, BaseSignal) and cls is not BaseSignal:
                            # Instantiate with default parameters
                            signal_instance = cls()
                            self.register(signal_instance)
                except Exception as module_err:
                    logger.error(f"Error importing module {full_module_name}: {module_err}")
                    
        except Exception as e:
            logger.error(f"Failed to dynamically discover signals: {e}")

    def compute_all(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all registered signals/features.
        """
        if price_df.empty:
            logger.warning("Empty price dataframe passed to feature computation.")
            return pd.DataFrame()

        # We must group by symbol and compute features for each ticker
        grouped = price_df.groupby("symbol")
        
        all_symbol_features = []
        
        for symbol, group in grouped:
            # Sort group by date chronologically
            symbol_df = group.sort_values("date").copy()
            
            # The base frame to merge other signals onto
            symbol_features_df = pd.DataFrame({
                "symbol": symbol_df["symbol"],
                "date": symbol_df["date"]
            })
            
            # Apply each signal
            for signal_name, signal in self._signals.items():
                try:
                    # Verify required columns are present in price_df
                    missing_cols = [c for c in signal.required_columns if c not in symbol_df.columns]
                    if missing_cols:
                        logger.warning(f"Ticker {symbol} missing columns {missing_cols} required by {signal_name}. Skipping signal.")
                        continue
                        
                    # Compute signal DataFrame
                    computed_signal = signal.compute(symbol_df)
                    
                    # Merge back onto symbol_features_df
                    symbol_features_df = pd.merge(
                        symbol_features_df, 
                        computed_signal, 
                        on=["symbol", "date"], 
                        how="left"
                    )
                except Exception as signal_err:
                    logger.error(f"Error computing signal '{signal_name}' for symbol '{symbol}': {signal_err}")
                    continue
            
            all_symbol_features.append(symbol_features_df)
            
        if not all_symbol_features:
            return pd.DataFrame()
            
        final_features_df = pd.concat(all_symbol_features, ignore_index=True)
        return final_features_df

    def run_pipeline(self) -> int:
        """
        Executes the entire feature engineering pipeline:
        1. Reads prices from DuckDB.
        2. Computes features.
        3. Migrates DB columns if necessary.
        4. Upserts to signals table.
        """
        logger.info("Executing feature engineering pipeline...")
        
        # 1. Fetch pricing data
        with get_db_connection(read_only=True) as conn:
            # Check if daily_prices exists
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
            
        # 2. Compute features
        features_df = self.compute_all(price_df)
        if features_df.empty:
            logger.warning("No features were computed.")
            return 0
            
        # 3. Handle DB persistence and automatic schema migration
        logger.info(f"Computed {len(features_df)} feature rows. Saving to DuckDB...")
        
        # Columns to write
        # Exclude metadata like symbol and date, which are primary keys
        feature_cols = [col for col in features_df.columns if col not in ["symbol", "date"]]
        
        try:
            with get_db_connection(read_only=False) as conn:
                # Initialize DB tables (creates signals table with defaults if not exists)
                init_db(conn)
                
                # Dynamic column checking and migration
                db_cols_result = conn.execute("PRAGMA table_info('signals')").fetchall()
                db_col_names = [row[1] for row in db_cols_result]
                
                # Check if features_df has columns that aren't in DuckDB signals table
                for col in feature_cols:
                    if col not in db_col_names:
                        logger.info(f"Database Migration: Adding missing column '{col}' to 'signals' table.")
                        conn.execute(f"ALTER TABLE signals ADD COLUMN {col} DOUBLE")
                        
                # Bulk upsert the features dataframe using DuckDB's native pandas integration
                # We dynamically construct the insert columns list since new features might have been added
                all_cols_in_df = ["symbol", "date"] + feature_cols
                cols_str = ", ".join(all_cols_in_df)
                select_str = ", ".join(all_cols_in_df)
                
                conn.execute(f"""
                    INSERT OR REPLACE INTO signals ({cols_str})
                    SELECT {select_str} FROM features_df
                """)
                
                # Get total rows
                total_rows = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
                
            logger.info(f"Successfully computed and saved features. Total rows in 'signals' table: {total_rows}")
            return len(features_df)
            
        except Exception as e:
            logger.error(f"Failed to persist computed features to DuckDB: {e}")
            raise
