from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Defines the standard interface for generating entry and exit signals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_signals(self, data_dict: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates entry (buy) and exit (sell) signal masks.

        Parameters:
        -----------
        data_dict : dict[str, pd.DataFrame]
            A dictionary where keys are data or indicator names (e.g. 'close', 'rsi')
            and values are pivoted DataFrames where index is Date and columns are Symbols.

        Returns:
        --------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing:
            1. entries: Boolean DataFrame where True indicates entry (buy) trigger.
            2. exits: Boolean DataFrame where True indicates exit (sell) trigger.
        """
        pass

class RSIStrategy(BaseStrategy):
    """
    Mean-reversion strategy based on the Relative Strength Index (RSI).
    - Buy (Entry) when RSI cross below or is below lower threshold (default: 30)
    - Sell (Exit) when RSI crosses above or is above upper threshold (default: 70)
    """

    def __init__(self, entry_threshold: float = 30.0, exit_threshold: float = 70.0):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    @property
    def name(self) -> str:
        return f"RSI_Strategy_{int(self.entry_threshold)}_{int(self.exit_threshold)}"

    def generate_signals(self, data_dict: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates entry and exit signals for RSI.
        """
        if "rsi" not in data_dict:
            raise ValueError("RSI strategy requires 'rsi' DataFrame in the data dictionary.")
            
        rsi = data_dict["rsi"]
        
        # Sells when RSI is above exit threshold
        entries = rsi < self.entry_threshold
        exits = rsi > self.exit_threshold
        
        return entries, exits
