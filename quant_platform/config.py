import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "quant_platform.db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default Ingestion Settings
DEFAULT_START_DATE = "2024-01-01"

# A representative subset of S&P 500 for testing or quick ingestion
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "TSLA", "NVDA", "JPM", "V", "PG",
    "UNH", "HD", "DIS", "PYPL", "NFLX"
]

# S&P 500 Tickers Source URL (Wikipedia is standard)
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Central Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
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
