import numpy as np
import pandas as pd

from quant_platform.features.base import BaseSignal


class VolatilitySignal(BaseSignal):
    """
    Computes a rolling close-price volatility ratio.

    Two variants are produced so both the "dispersion" and the "normalised"
    views are available:

    * ``volatility_15d``     - 15-day rolling standard deviation of close.
    * ``volatility_ratio_15d`` - that standard deviation divided by close.
    """

    def __init__(self, window: int = 15):
        if window < 2:
            raise ValueError("window must be at least 2 to compute a standard deviation.")
        self._window = window

    @property
    def name(self) -> str:
        return "custom_volatility"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.sort_values("date").copy()

        result_df = pd.DataFrame({
            "symbol": df["symbol"],
            "date": df["date"],
        })

        rolling_std = df["close"].rolling(window=self._window).std()

        result_df["volatility_15d"] = rolling_std
        result_df["volatility_ratio_15d"] = rolling_std / df["close"].replace(0, np.nan)

        return result_df
