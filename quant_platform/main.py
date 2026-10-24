import argparse
import logging
import logging.config
import sys

from quant_platform import config
from quant_platform.config import (
    BACKTEST_RESULTS_PATH,
    DEFAULT_EXECUTION_DELAY,
    DEFAULT_FEE,
    DEFAULT_INIT_CASH,
    DEFAULT_START_DATE,
    LOGGING_CONFIG,
)
from quant_platform.data_access import require_tables
from quant_platform.db.connection import get_db_connection
from quant_platform.db.identifiers import quote_identifier
from quant_platform.features.factory import SignalFactory
from quant_platform.ingestion.pipeline import IngestionPipeline
from quant_platform.ingestion.yfinance_fetcher import YFinanceFetcher

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
        logger.info("Custom tickers specified: %s", tickers)
    elif args.test:
        tickers = list(config.DEFAULT_TICKERS)
        logger.info("Running in TEST mode with a representative subset: %s", tickers)

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
        logger.info("Ingestion process completed. Total new rows downloaded: %d", rows_ingested)
    except Exception:
        logger.exception("Failed to complete ingestion.")
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
        logger.info("Feature engineering pipeline completed. Total rows calculated: %d", features_computed)
    except Exception:
        logger.exception("Failed to complete feature engineering.")
        sys.exit(1)


def handle_status(args):
    """
    Handles the 'status' CLI command.
    Prints database statistics and checks health.
    """
    logger.info("Checking Quantitative Platform Status...")

    try:
        require_tables()
    except ValueError as e:
        print(f"Status unavailable: {e}")
        return

    try:
        _print_status()
    except Exception:
        logger.exception("Failed to read database status.")
        sys.exit(1)


def _print_status():
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

            # The sample query is built from the discovered feature columns so
            # it keeps working when the signal set changes.
            sample_cols = feature_cols[:3]
            if sample_cols:
                selected = ", ".join(["symbol", "date", *(quote_identifier(c) for c in sample_cols)])
                non_null = quote_identifier(sample_cols[0])
                sample_features = conn.execute(
                    f"SELECT {selected} FROM signals WHERE {non_null} IS NOT NULL LIMIT 5"
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
        logger.error("Strategy '%s' is not supported yet.", args.strategy)
        sys.exit(1)

    # 2. Instantiate and run Backtest Engine
    try:
        from quant_platform.backtest.engine import BacktestEngine

        engine = BacktestEngine(strategy, execution_delay=args.execution_delay)
        engine.run(
            init_cash=args.init_cash,
            fee=args.fee,
            slippage=args.slippage,
            output_path=args.output,
            start=args.start_date,
            end=args.end_date,
            symbols=[s.strip() for s in args.symbols.split(",")] if args.symbols else None,
        )
    except Exception:
        logger.exception("Failed to complete backtest run.")
        sys.exit(1)


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be positive")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be >= 0")
    return number


def _execution_delay(value: str) -> int:
    delay = int(value)
    if delay < 1:
        raise argparse.ArgumentTypeError(
            "execution delay must be >= 1 so a signal computed from a bar's close cannot fill at that same close"
        )
    return delay


def main():
    # Configure logging at the entry point only, so importing this module
    # never reconfigures the host application's loggers.
    logging.config.dictConfig(LOGGING_CONFIG)

    parser = argparse.ArgumentParser(description="Quant Research Platform Ingestion & Feature Engineering CLI")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available Commands")

    # 1. Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest daily price & volume data from yfinance")
    ticker_source = ingest_parser.add_mutually_exclusive_group()
    ticker_source.add_argument(
        "--tickers", type=str, help="Comma-separated list of symbols (e.g. AAPL,MSFT,SPY). If omitted, fetches S&P 500."
    )
    ticker_source.add_argument(
        "--test", action="store_true", help="Ingest only a small representative test subset of tickers (faster)"
    )
    ingest_parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date in YYYY-MM-DD format (default: {DEFAULT_START_DATE})",
    )
    ingest_parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format, exclusive (default: tomorrow, so today's session is included)",
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
        type=_positive_float,
        default=DEFAULT_INIT_CASH,
        help=f"Total initial capital in cash, split across symbols (default: {DEFAULT_INIT_CASH})",
    )
    backtest_parser.add_argument(
        "--fee",
        type=_positive_float,
        default=DEFAULT_FEE,
        help=f"Transaction fee per trade as a fraction (default: {DEFAULT_FEE}, i.e. 10 bps)",
    )
    backtest_parser.add_argument(
        "--slippage", type=_non_negative_float, default=0.0, help="Slippage per trade as a fraction (default: 0.0)"
    )
    backtest_parser.add_argument(
        "--execution-delay",
        type=_execution_delay,
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
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Backtest window start in YYYY-MM-DD format, inclusive (default: full history).",
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Backtest window end in YYYY-MM-DD format, inclusive (default: full history).",
    )
    backtest_parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated subset of symbols to backtest (default: all ingested symbols).",
    )

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
