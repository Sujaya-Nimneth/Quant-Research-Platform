from abc import ABC, abstractmethod

import pandas as pd


class BaseFetcher(ABC):
    """
    Abstract base class for all pricing data fetchers.
    Any new data provider (e.g., Alpaca, Alpha Vantage, Interactive Brokers)
    should inherit from this class and implement the abstract methods.
    """

    @abstractmethod
    def fetch_daily_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily historical pricing and volume data for a list of tickers.

        Parameters:
        -----------
        tickers : list[str]
            List of ticker symbols to fetch data for.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing daily pricing data with the following columns:
            ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            The 'date' column must be of type datetime64 or date.
        """
        pass

    def fetch_sp500_tickers(self, strict: bool = False) -> list[str]:
        """
        Discover the current S&P 500 constituent list.

        Fetchers without universe discovery raise ``NotImplementedError`` and
        the pipeline falls back to ``config.DEFAULT_TICKERS``.

        Parameters:
        -----------
        strict : bool
            Fail instead of silently returning a partial list.
        """
        raise NotImplementedError("This fetcher does not support S&P 500 ticker discovery.")
