import argparse
import logging.config
import sys

from quant_platform.config import (
    BACKTEST_RESULTS_PATH,
    DEFAULT_EXECUTION_DELAY,
    DEFAULT_FEE,
    DEFAULT_START_DATE,
    LOGGING_CONFIG,
)
from quant_platform.db.connection import get_db_connection
from quant_platform.features.factory import SignalFactory
from quant_platform.ingestion.pipeline import IngestionPipeline
from quant_platform.ingestion.yfinance_fetcher import YFinanceFetcher

# Initialize central logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("quant_platform")


def handle_ingest(args):
    """
    Handles the 'ingest' CLI command.
    """
    logger.info("Initializing Data Ingestion Inbound Pipeline...")

    # Parse tickers
    tickers = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
        logger.info(f"Custom tickers specified: {tickers}")
    elif args.test:
        from quant_platform.config import DEFAULT_TICKERS

        tickers = DEFAULT_TICKERS
        logger.info(f"Running in TEST mode with a representative subset: {tickers}")

    # Fetcher and Pipeline
    fetcher = YFinanceFetcher()
    pipeline = IngestionPipeline(fetcher)

    # Run
    try:
        rows_ingested = pipeline.run(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            full_refresh=args.full_refresh,
            strict_tickers=args.strict_tickers,
        )
        logger.info(f"Ingestion process completed. Total new rows downloaded: {rows_ingested}")
    except Exception as e:
        logger.error(f"Failed to complete ingestion: {e}")
        sys.exit(1)


def handle_features(args):
    """
    Handles the 'features' CLI command.
    """
    logger.info("Initializing Signal Factory & Feature Engineering Pipeline...")

    # Run
    try:
        factory = SignalFactory()
        features_computed = factory.run_pipeline()
        logger.info(f"Feature engineering pipeline completed. Total rows calculated: {features_computed}")
    except Exception as e:
        logger.error(f"Failed to complete feature engineering: {e}")
        sys.exit(1)


def handle_status(args):
    """
    Handles the 'status' CLI command.
    Prints database statistics and checks health.
    """
    logger.info("Checking Quantitative Platform Status...")

    with get_db_connection(read_only=True) as conn:
        # Check tables
        tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
        table_names = [t[0] for t in tables]

        print("\n" + "=" * 50)
        print("         QUANT PLATFORM DATABASE DIAGNOSTICS")
        print("=" * 50)
        print(f"Active Tables: {', '.join(table_names) if table_names else 'None'}\n")

        if "daily_prices" in table_names:
            prices_count = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
            symbols_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM daily_prices").fetchone()[0]
            date_range = conn.execute("SELECT MIN(date), MAX(date) FROM daily_prices").fetchone()

            print("Table: daily_prices")
            print(f"  - Total pricing records: {prices_count:,}")
            print(f"  - Unique tickers:         {symbols_count}")
            print(f"  - Historical date range:  {date_range[0]} to {date_range[1]}")

            # Print sample tickers
            sample_tickers = conn.execute(
                "SELECT symbol, COUNT(*) FROM daily_prices GROUP BY symbol LIMIT 5"
            ).fetchall()
            print("  - Sample Tickers:")
            for symbol, cnt in sample_tickers:
                print(f"    - {symbol:<5}: {cnt:,} daily records")
            print()
        else:
            print("Table: daily_prices [NOT CREATED YET]\n")

        if "signals" in table_names:
            signals_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            signals_cols = conn.execute("PRAGMA table_info('signals')").fetchall()
            col_names = [row[1] for row in signals_cols]

            # Metadata columns are excluded by name, not by position: signal
            # modules can add columns at any time via schema migration.
            metadata_cols = {"symbol", "date", "calculated_at"}
            feature_cols = [c for c in col_names if c not in metadata_cols]

            print("Table: signals")
            print(f"  - Total features records: {signals_count:,}")
            print(f"  - Computed Alpha features: {', '.join(feature_cols)}")

            # Print a few rows of sample calculated features
            sample_features = conn.execute(
                "SELECT symbol, date, macd_line, rsi, bb_percent FROM signals WHERE macd_line IS NOT NULL LIMIT 5"
            ).df()

            if not sample_features.empty:
                print("  - Sample Feature Vectors:")
                print(sample_features.to_string(index=False))
            print()
        else:
            print("Table: signals [NOT CREATED YET]\n")
        print("=" * 50 + "\n")


def handle_backtest(args):
    """
    Handles the 'backtest' CLI command.
    """
    logger.info("Initializing Backtesting CLI Module...")

    # 1. Instantiate selected strategy
    if args.strategy == "rsi":
        from quant_platform.backtest.strategy import RSIStrategy

        strategy = RSIStrategy(entry_threshold=args.rsi_entry, exit_threshold=args.rsi_exit)
    else:
        logger.error(f"Strategy '{args.strategy}' is not supported yet.")
        sys.exit(1)

    # 2. Instantiate and run Backtest Engine
    try:
        from quant_platform.backtest.engine import BacktestEngine

        engine = BacktestEngine(strategy, execution_delay=args.execution_delay)
        engine.run(init_cash=args.init_cash, fee=args.fee, slippage=args.slippage, output_path=args.output)
    except Exception as e:
        logger.error(f"Failed to complete backtest run: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Quant Research Platform Ingestion & Feature Engineering CLI")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available Commands")

    # 1. Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest daily price & volume data from yfinance")
    ingest_parser.add_argument(
        "--tickers", type=str, help="Comma-separated list of symbols (e.g. AAPL,MSFT,SPY). If omitted, fetches S&P 500."
    )
    ingest_parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date in YYYY-MM-DD format (default: {DEFAULT_START_DATE})",
    )
    ingest_parser.add_argument("--end-date", type=str, help="End date in YYYY-MM-DD format (default: today)")
    ingest_parser.add_argument(
        "--test", action="store_true", help="Ingest only a small representative test subset of tickers (faster)"
    )
    ingest_parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-download the entire history instead of only new sessions (default: incremental)",
    )
    ingest_parser.add_argument(
        "--strict-tickers",
        action="store_true",
        help="Fail if the S&P 500 constituent list cannot be fetched instead of using the 15-ticker default",
    )

    # 2. Features Command
    subparsers.add_parser("features", help="Execute Signal Factory to compute features & save to DuckDB")

    # 3. Status Command
    subparsers.add_parser("status", help="Print database diagnostics and sample feature rows")

    # 4. Backtest Command
    backtest_parser = subparsers.add_parser(
        "backtest", help="Simulate a portfolio using VectorBT signal-based backtesting"
    )
    backtest_parser.add_argument(
        "--strategy", type=str, default="rsi", choices=["rsi"], help="Name of the strategy to run (default: rsi)"
    )
    backtest_parser.add_argument(
        "--init-cash",
        type=float,
        default=10000.0,
        help="Total initial capital in cash, split across symbols (default: 10000.0)",
    )
    backtest_parser.add_argument(
        "--fee",
        type=float,
        default=DEFAULT_FEE,
        help=f"Transaction fee per trade as a fraction (default: {DEFAULT_FEE}, i.e. 10 bps)",
    )
    backtest_parser.add_argument(
        "--slippage", type=float, default=0.0, help="Slippage per trade as a fraction (default: 0.0)"
    )
    backtest_parser.add_argument(
        "--execution-delay",
        type=int,
        default=DEFAULT_EXECUTION_DELAY,
        help=(
            "Bars to wait before filling a signal (default: "
            f"{DEFAULT_EXECUTION_DELAY}). Must be >= 1 to avoid look-ahead bias."
        ),
    )
    backtest_parser.add_argument(
        "--output",
        type=str,
        default=str(BACKTEST_RESULTS_PATH),
        help=f"Target JSON path to export trade logs & statistics (default: {BACKTEST_RESULTS_PATH})",
    )
    backtest_parser.add_argument("--rsi-entry", type=float, default=30.0, help="RSI buy threshold (default: 30.0)")
    backtest_parser.add_argument("--rsi-exit", type=float, default=70.0, help="RSI sell threshold (default: 70.0)")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "features":
        handle_features(args)
    elif args.command == "status":
        handle_status(args)
    elif args.command == "backtest":
        handle_backtest(args)


if __name__ == "__main__":
    main()
