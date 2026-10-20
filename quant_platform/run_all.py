"""Sequential end-to-end pipeline runner: ingest -> features -> backtest."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Allow `python quant_platform/run_all.py` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_platform.config import (  # noqa: E402
    BACKTEST_RESULTS_PATH,
    BASE_DIR,
    DEFAULT_FEE,
    DEFAULT_INIT_CASH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] quant_platform_runner: %(message)s",
)
logger = logging.getLogger("quant_platform_runner")


def _run_step(label: str, args: list[str], echo_stdout: bool = False) -> None:
    """Run one pipeline stage as a subprocess.

    The child inherits ``BASE_DIR`` as its working directory and on
    ``sys.path``, so the runner works regardless of where it is invoked from
    (cron, systemd, an IDE task, or another directory).
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(BASE_DIR), env.get("PYTHONPATH", "")])
    )
    command = [sys.executable, "-m", *args]
    logger.info("Running: %s", " ".join(command))

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE_DIR),
            env=env,
        )
    except subprocess.CalledProcessError as e:
        logger.error("FATAL: %s failed with exit code %d!", label, e.returncode)
        if e.stdout:
            logger.error("--- stdout ---\n%s", e.stdout.rstrip())
        if e.stderr:
            logger.error("--- stderr ---\n%s", e.stderr.rstrip())
        raise SystemExit(1) from e

    if result.stdout:
        if echo_stdout:
            print(result.stdout, end="")
        else:
            logger.info(result.stdout.rstrip())
    if result.stderr:
        logger.debug(result.stderr.rstrip())

    logger.info("%s completed successfully.", label)


def run_all(
    test_mode: bool = True,
    full_refresh: bool = False,
    init_cash: float = DEFAULT_INIT_CASH,
    fee: float = DEFAULT_FEE,
) -> None:
    """
    Executes the entire quantitative pipeline sequentially:
    1. Ingestion Pipeline
    2. Signal Factory / Feature Engineering
    3. VectorBT Backtest Simulation
    """
    logger.info("=" * 50)
    logger.info("   LAUNCHING QUANTITATIVE PLATFORM DAILY PIPELINE")
    logger.info("=" * 50)

    # 1. Ingestion
    logger.info("Step 1/3: Running yfinance market data ingestion...")
    ingest_args = ["quant_platform.main", "ingest"]
    if test_mode:
        ingest_args.append("--test")
        logger.info("Running in TEST mode (15 representative S&P 500 tickers).")
    if full_refresh:
        ingest_args.append("--full-refresh")
    if not test_mode:
        ingest_args.append("--strict-tickers")
    _run_step("Data ingestion", ingest_args)

    # 2. Features
    logger.info("Step 2/3: Recalculating alpha signals & indicators...")
    _run_step("Signal recalculation", ["quant_platform.main", "features"])

    # 3. Backtest
    logger.info("Step 3/3: Simulating RSI mean-reversion backtest...")
    _run_step(
        "Backtest simulation",
        [
            "quant_platform.main", "backtest",
            "--strategy", "rsi",
            "--init-cash", str(init_cash),
            "--fee", str(fee),
            "--output", str(BACKTEST_RESULTS_PATH),
        ],
        echo_stdout=True,
    )

    logger.info("=" * 50)
    logger.info("   DAILY PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full quant platform pipeline.")
    parser.add_argument(
        "--full-universe",
        action="store_true",
        help="Ingest the full S&P 500 instead of the 15-ticker test subset.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-download all history instead of only new sessions.",
    )
    parser.add_argument("--init-cash", type=float, default=DEFAULT_INIT_CASH)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    args = parser.parse_args()

    run_all(
        test_mode=not args.full_universe,
        full_refresh=args.full_refresh,
        init_cash=args.init_cash,
        fee=args.fee,
    )


if __name__ == "__main__":
    main()
