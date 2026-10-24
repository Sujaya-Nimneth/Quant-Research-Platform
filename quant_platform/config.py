from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "quant_platform.db"
BACKTEST_RESULTS_PATH = DATA_DIR / "backtest_results.json"


def ensure_dirs() -> None:
    """Create the data directory on demand.

    Kept out of module import so importing the package never writes to the
    filesystem; entry points that write data call this first.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Default Ingestion Settings
DEFAULT_START_DATE = "2024-01-01"

# A representative subset of S&P 500 for testing or quick ingestion
DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "NVDA",
    "JPM",
    "V",
    "PG",
    "UNH",
    "HD",
    "DIS",
    "PYPL",
    "NFLX",
]

# S&P 500 Tickers Source URL (Wikipedia is standard)
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Central Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

# --- Backtest defaults -------------------------------------------------------
DEFAULT_INIT_CASH = 10_000.0
DEFAULT_FEE = 0.001

# Signals are shifted by this many bars before execution so a signal computed
# from a bar's close cannot be filled at that same close (look-ahead bias).
DEFAULT_EXECUTION_DELAY = 1
