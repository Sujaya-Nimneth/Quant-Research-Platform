import pandas as pd
import numpy as np
from quant_platform.features.base import BaseSignal

class MeanReversionSignals(BaseSignal):
    """
    Computes two standard mean-reversion indicators:
    1. RSI (Relative Strength Index): 14-period standard index.
    2. Bollinger Bands: 20-period standard deviation bands and %B.
    """

    def __init__(self, rsi_period: int = 14, bb_period: int = 20, bb_std: float = 2.0):
        self._rsi_period = rsi_period
        self._bb_period = bb_period
        self._bb_std = bb_std

    @property
    def name(self) -> str:
        return "mean_reversion_signals"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes RSI and Bollinger Bands for the given pricing DataFrame.
        Expected input DataFrame contains: ['symbol', 'date', 'close']
        """
        # Ensure data is sorted by date
        df = data.sort_values("date").copy()
        
        result_df = pd.DataFrame({
            "symbol": df["symbol"],
            "date": df["date"]
        })

        close = df["close"]

        # 1. RSI (Relative Strength Index) using Wilder's EMA smoothing technique
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Wilder's smoothing is equivalent to an exponential moving average with alpha = 1 / period
        # ewm(alpha=1/period, adjust=False) is standard for wilder's
        alpha = 1.0 / self._rsi_period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        # We need to seed the first value with SMA, but yfinance standard ewm already handles initial values gracefully.
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # If avg_loss was 0, RSI should be 100 if avg_gain > 0, else 50
        rsi = rsi.fillna(100.0)
        # Ensure first period is NaN since diff is NaN
        rsi.iloc[0] = np.nan
        
        result_df["rsi"] = rsi

        # 2. Bollinger Bands (20-day Simple Moving Average +/- 2 Standard Deviations)
        sma = close.rolling(window=self._bb_period).mean()
        std = close.rolling(window=self._bb_period).std()
        
        bb_upper = sma + (self._bb_std * std)
        bb_lower = sma - (self._bb_std * std)
        
        # %B indicates where the price is relative to the bands
        # Avoid division by zero
        band_diff = bb_upper - bb_lower
        bb_percent = (close - bb_lower) / band_diff.replace(0, np.nan)
        bb_percent = bb_percent.fillna(0.5)  # Default if bands collapse to zero

        result_df["bb_upper"] = bb_upper
        result_df["bb_lower"] = bb_lower
        result_df["bb_percent"] = bb_percent

        return result_df
