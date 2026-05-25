import pandas as pd
from quant_platform.features.base import BaseSignal

class MomentumSignals(BaseSignal):
    """
    Computes three standard momentum indicators:
    1. MACD (Moving Average Convergence Divergence): MACD line, Signal line, MACD histogram.
    2. ROC (Rate of Change): N-period percent difference.
    3. Momentum: Ratio of Close price to its N-period Simple Moving Average.
    """

    def __init__(self, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9, roc_period: int = 12, mom_period: int = 10):
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._roc_period = roc_period
        self._mom_period = mom_period

    @property
    def name(self) -> str:
        return "momentum_signals"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes MACD, ROC, and SMA Momentum for the given pricing DataFrame.
        Expected input DataFrame contains: ['symbol', 'date', 'close']
        """
        # Ensure data is sorted by date
        df = data.sort_values("date").copy()
        
        result_df = pd.DataFrame({
            "symbol": df["symbol"],
            "date": df["date"]
        })

        close = df["close"]

        # 1. MACD (Moving Average Convergence Divergence)
        ema_fast = close.ewm(span=self._macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self._macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal_line = macd_line.ewm(span=self._macd_signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal_line

        result_df["macd_line"] = macd_line
        result_df["macd_signal"] = macd_signal_line
        result_df["macd_hist"] = macd_hist

        # 2. ROC (Rate of Change)
        shifted_close_roc = close.shift(self._roc_period)
        # Avoid division by zero
        result_df["roc"] = ((close - shifted_close_roc) / shifted_close_roc) * 100.0

        # 3. Momentum (Ratio of Close to its Simple Moving Average)
        sma = close.rolling(window=self._mom_period).mean()
        result_df["momentum"] = close / sma

        return result_df
