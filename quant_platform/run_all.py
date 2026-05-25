import subprocess
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] quant_platform_runner: %(message)s"
)
logger = logging.getLogger("quant_platform_runner")

def run_all(test_mode: bool = True):
    """
    Executes the entire quantitative pipeline sequentially:
    1. Ingestion Pipeline
    2. Signal Factory / Feature Engineering
    3. VectorBT Backtest Simulation
    """
    logger.info("==================================================")
    logger.info("   LAUNCHING QUANTITATIVE PLATFORM DAILY PIPELINE")
    logger.info("==================================================")
    
    # 1. Ingestion
    logger.info("Step 1/3: Running yfinance Market Data Ingestion...")
    ingest_cmd = [sys.executable, "-m", "quant_platform.main", "ingest"]
    if test_mode:
        ingest_cmd.append("--test")
        logger.info("Running in TEST mode (15 representative S&P 500 tickers)...")
    
    try:
        result = subprocess.run(ingest_cmd, capture_output=True, text=True, check=True)
        logger.info("Data ingestion completed successfully.")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"FATAL: Data Ingestion failed with exit code {e.returncode}!")
        logger.error(e.stderr)
        sys.exit(1)
        
    # 2. Features
    logger.info("Step 2/3: Recalculating Alpha Signals & Indicators...")
    features_cmd = [sys.executable, "-m", "quant_platform.main", "features"]
    try:
        result = subprocess.run(features_cmd, capture_output=True, text=True, check=True)
        logger.info("Signal recalculation completed successfully.")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"FATAL: Signal generation failed with exit code {e.returncode}!")
        logger.error(e.stderr)
        sys.exit(1)
        
    # 3. Backtest
    logger.info("Step 3/3: Simulating RSI Mean-Reversion Backtest...")
    backtest_cmd = [
        sys.executable, "-m", "quant_platform.main", "backtest",
        "--strategy", "rsi",
        "--init-cash", "10000.0",
        "--fee", "0.001"
    ]
    try:
        result = subprocess.run(backtest_cmd, capture_output=True, text=True, check=True)
        logger.info("Backtest simulation completed successfully.")
        logger.info(result.stdout)  # Print the performance tear sheet directly!
    except subprocess.CalledProcessError as e:
        logger.error(f"FATAL: Backtest simulation failed with exit code {e.returncode}!")
        logger.error(e.stderr)
        sys.exit(1)
        
    logger.info("==================================================")
    logger.info("   DAILY PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("==================================================")

if __name__ == "__main__":
    # Default to test mode to avoid long S&P 500 runs in scheduler unless customized
    run_all(test_mode=True)
