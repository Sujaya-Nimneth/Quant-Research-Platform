import pandas as pd
from quant_platform.features.base import BaseSignal

class VolatilitySignal(BaseSignal):
    """
    Computes custom close price volatility ratio over a rolling 15-day window.
    """
    
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
            "date": df["date"]
        })
        
        # Calculate 15-day rolling volatility
        rolling_std = df["close"].rolling(window=15).std()
        
        # Calculate volatility ratio
        result_df["volatility_15d"] = rolling_std / df["close"]
        
        return result_df
