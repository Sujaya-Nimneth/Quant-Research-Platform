import numpy as np
import pandas as pd

from quant_platform.features.base import BaseSignal


class MomentumSignals(BaseSignal):
    """
    Computes three standard momentum indicators:
    1. MACD (Moving Average Convergence Divergence): MACD line, Signal line, histogram.
    2. ROC (Rate of Change): N-period percent change.
    3. Price-to-SMA: close divided by its N-period simple moving average.

    Warmup rows are NaN rather than zero-filled, and ratios that would divide
    by zero are NaN rather than +/-inf so downstream code never sees infinities.
    """

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        roc_period: int = 12,
        mom_period: int = 10,
    ):
        if macd_fast >= macd_slow:
            raise ValueError(f"macd_fast ({macd_fast}) must be less than macd_slow ({macd_slow}).")
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
        Computes MACD, ROC, and price-to-SMA for the given pricing DataFrame.
        Expected input DataFrame contains: ['symbol', 'date', 'close']
        """
        df = data.sort_values("date").copy()

        result_df = pd.DataFrame(
            {
                "symbol": df["symbol"],
                "date": df["date"],
            }
        )

        close = df["close"]

        # 1. MACD
        ema_fast = close.ewm(span=self._macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self._macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal_line = macd_line.ewm(span=self._macd_signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal_line

        # EWM emits values from bar 0, but they are statistically meaningless
        # until the slow EMA has converged and the signal line has a window;
        # mask the warmup so no strategy can trade on it.
        macd_warmup = self._macd_slow + self._macd_signal
        macd_line.iloc[:macd_warmup] = np.nan
        macd_signal_line.iloc[:macd_warmup] = np.nan
        macd_hist.iloc[:macd_warmup] = np.nan

        result_df["macd_line"] = macd_line
        result_df["macd_signal"] = macd_signal_line
        result_df["macd_hist"] = macd_hist

        # 2. ROC, expressed in percent. Guard against a zero prior close.
        prior_close = close.shift(self._roc_period)
        safe_prior = prior_close.replace(0, np.nan)
        result_df["roc"] = ((close - safe_prior) / safe_prior) * 100.0

        # 3. Price relative to its simple moving average (1.0 == at the SMA).
        sma = close.rolling(window=self._mom_period).mean().replace(0, np.nan)
        result_df["momentum"] = close / sma

        return result_df
