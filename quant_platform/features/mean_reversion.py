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

        avg_gain = _wilder_average(gain, period)
        avg_loss = _wilder_average(loss, period)

        rsi = pd.Series(np.nan, index=close.index, dtype=float)

        valid = avg_gain.notna() & avg_loss.notna()
        # No losses at all: RSI is 100 when there are gains, 50 when flat.
        no_loss = valid & (avg_loss == 0)
        rsi[no_loss & (avg_gain > 0)] = 100.0
        rsi[no_loss & (avg_gain == 0)] = 50.0

        normal = valid & (avg_loss > 0)
        rs = avg_gain[normal] / avg_loss[normal]
        rsi[normal] = 100.0 - (100.0 / (1.0 + rs))

        return rsi


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed average of ``values`` (seed = SMA, then EMA recursion).

    Computed per contiguous run of non-NaN values so a data gap re-seeds the
    smoothing instead of poisoning the rest of the series. Within a run, the
    SMA of the first `period` values seeds an ``ewm(alpha=1/period)`` chain,
    which is algebraically Wilder's recursion but runs at C speed.
    """
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna().to_numpy()

    # Split the series into contiguous runs of valid values.
    positions = np.arange(len(values))
    run_id = np.cumsum(~valid)
    valid_positions = positions[valid]
    valid_run_ids = run_id[valid]
    boundaries = np.flatnonzero(np.diff(valid_run_ids) != 0) + 1
    for run_positions in np.split(valid_positions, boundaries):
        run = values.iloc[run_positions]
        if len(run) < period:
            continue

        seed = float(run.iloc[:period].mean())
        # ewm over [seed, rest...] reproduces y[k] = (1-a)*y[k-1] + a*x[k]
        # exactly, starting from the SMA seed.
        chain = pd.concat([pd.Series([seed], dtype=float), run.iloc[period:]], ignore_index=True)
        smoothed = chain.ewm(alpha=1.0 / period, adjust=False).mean()

        out.iloc[run_positions[period - 1 :]] = smoothed.to_numpy()

    return out
