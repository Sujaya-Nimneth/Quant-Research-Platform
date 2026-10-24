import numpy as np
import pandas as pd

from quant_platform.features.base import BaseSignal


class MeanReversionSignals(BaseSignal):
    """
    Computes two standard mean-reversion indicators:
    1. RSI (Relative Strength Index): 14-period, Wilder's smoothing.
    2. Bollinger Bands: 20-period standard deviation bands and %B.

    Values inside the rolling warmup window are left as NaN. Emitting a
    placeholder (100 for RSI, 0.5 for %B) would look like a real signal and
    generate trades before the indicator has enough history to be meaningful.
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
        df = data.sort_values("date").copy()

        result_df = pd.DataFrame(
            {
                "symbol": df["symbol"],
                "date": df["date"],
            }
        )

        close = df["close"]

        result_df["rsi"] = self._rsi(close)

        # Bollinger Bands (20-day SMA +/- 2 standard deviations). Population
        # std (ddof=0) matches the classic definition and TA-Lib.
        sma = close.rolling(window=self._bb_period).mean()
        std = close.rolling(window=self._bb_period).std(ddof=0)

        bb_upper = sma + (self._bb_std * std)
        bb_lower = sma - (self._bb_std * std)

        # %B: where price sits relative to the bands. A zero-width band (flat
        # price) is genuinely undefined, so leave it NaN instead of guessing.
        band_width = bb_upper - bb_lower
        bb_percent = (close - bb_lower) / band_width.replace(0, np.nan)

        result_df["bb_upper"] = bb_upper
        result_df["bb_lower"] = bb_lower
        result_df["bb_percent"] = bb_percent

        return result_df

    def _rsi(self, close: pd.Series) -> pd.Series:
        """Wilder's RSI.

        The first average gain/loss is a simple mean of the first `period`
        deltas; subsequent values use Wilder's recursive smoothing
        (equivalent to an EMA with alpha = 1/period). The result is NaN until
        a full window is available, and re-seeds the same way after a data
        gap instead of staying NaN for the rest of the series.
        """
        period = self._rsi_period
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Seed with the SMA of the first `period` observations.
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        seeded_gain = avg_gain.copy()
        seeded_loss = avg_loss.copy()

        # Wilder smoothing over the remaining observations. The first smoothed
        # value sits at index `period` (from the rolling seed); recursion starts
        # one bar later so it has a valid predecessor.
        gain_values = gain.to_numpy(dtype=float)
        loss_values = loss.to_numpy(dtype=float)
        alpha = 1.0 / period

        for i in range(len(close)):
            if np.isnan(gain_values[i]) or np.isnan(loss_values[i]):
                # Data gap: nothing to smooth. The next bars re-seed below
                # instead of the recursion staying dead for the rest of the
                # series.
                continue

            if i == 0 or np.isnan(seeded_gain.iloc[i - 1]) or np.isnan(seeded_loss.iloc[i - 1]):
                # Start of series or immediately after a gap: fall back to the
                # rolling seed, which is available once `period` clean deltas
                # have accumulated again.
                if not np.isnan(avg_gain.iloc[i]) and not np.isnan(avg_loss.iloc[i]):
                    seeded_gain.iloc[i] = avg_gain.iloc[i]
                    seeded_loss.iloc[i] = avg_loss.iloc[i]
                continue

            prev_gain = seeded_gain.iloc[i - 1]
            prev_loss = seeded_loss.iloc[i - 1]
            seeded_gain.iloc[i] = prev_gain + alpha * (gain_values[i] - prev_gain)
            seeded_loss.iloc[i] = prev_loss + alpha * (loss_values[i] - prev_loss)

        rsi = pd.Series(np.nan, index=close.index, dtype=float)

        valid = seeded_gain.notna() & seeded_loss.notna()
        # No losses at all: RSI is 100 when there are gains, 50 when flat.
        no_loss = valid & (seeded_loss == 0)
        rsi[no_loss & (seeded_gain > 0)] = 100.0
        rsi[no_loss & (seeded_gain == 0)] = 50.0

        normal = valid & (seeded_loss > 0)
        rs = seeded_gain[normal] / seeded_loss[normal]
        rsi[normal] = 100.0 - (100.0 / (1.0 + rs))

        return rsi
