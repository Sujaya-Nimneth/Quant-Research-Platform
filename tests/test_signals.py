"""Signal correctness: warmup handling, numerical guards, and reference parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.features.mean_reversion import MeanReversionSignals
from quant_platform.features.momentum import MomentumSignals
from quant_platform.features.volatility import VolatilitySignal


def _close_frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": "AAA",
        "date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
        "close": close.to_numpy(),
    })


def _reference_wilder_rsi(close: pd.Series, period: int = 14) -> np.ndarray:
    delta = close.diff()
    gain = delta.clip(lower=0).to_numpy(dtype=float)
    loss = (-delta.clip(upper=0)).to_numpy(dtype=float)

    out = np.full(len(close), np.nan)
    avg_gain = gain[1:period + 1].mean()
    avg_loss = loss[1:period + 1].mean()
    out[period] = _rsi_from(avg_gain, avg_loss)

    for i in range(period + 1, len(close)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def test_rsi_matches_reference_implementation():
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    result = MeanReversionSignals().compute(_close_frame(close))
    expected = _reference_wilder_rsi(close)

    np.testing.assert_allclose(result["rsi"].to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_rsi_warmup_is_nan_not_filled():
    rng = np.random.default_rng(11)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 120)))
    rsi = MeanReversionSignals().compute(_close_frame(close))["rsi"]

    assert rsi.iloc[:14].isna().all(), "the 14-bar warmup must be NaN"
    assert rsi.iloc[14:].notna().all()
    # The old implementation blanket-filled warmup NaNs with 100, which looked
    # like a permanent overbought reading at the start of every series.
    assert rsi.iloc[14] != pytest.approx(100.0)


def test_rsi_handles_flat_series():
    close = pd.Series([50.0] * 40)
    rsi = MeanReversionSignals().compute(_close_frame(close))["rsi"]
    assert (rsi.dropna() == 50.0).all()


def test_rsi_monotonic_rise_is_100():
    close = pd.Series(np.arange(1, 60, dtype=float))
    rsi = MeanReversionSignals().compute(_close_frame(close))["rsi"]
    assert rsi.dropna().iloc[-1] == pytest.approx(100.0)


def test_bollinger_warmup_preserved():
    close = pd.Series(np.linspace(100, 130, 60))
    result = MeanReversionSignals().compute(_close_frame(close))

    assert result["bb_percent"].iloc[:19].isna().all()
    assert result["bb_upper"].iloc[19:].notna().all()
    assert result["bb_lower"].iloc[19:].notna().all()
    assert not (result["bb_percent"].dropna() == 0.5).all()


def test_roc_and_momentum_never_produce_infinities():
    close = pd.Series([100.0] * 30 + [0.0] + [5.0] * 30)
    result = MomentumSignals().compute(_close_frame(close))

    numeric = result[["roc", "momentum", "macd_line"]].to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_momentum_roc_matches_pct_change():
    close = pd.Series(np.linspace(100, 150, 60))
    result = MomentumSignals(roc_period=5).compute(_close_frame(close))
    expected = close.pct_change(5) * 100.0
    np.testing.assert_allclose(
        result["roc"].to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9
    )


def test_momentum_rejects_inverted_macd_windows():
    with pytest.raises(ValueError):
        MomentumSignals(macd_fast=26, macd_slow=12)


def test_volatility_ratio_is_scale_free():
    close = pd.Series(np.linspace(100, 200, 60))
    result = VolatilitySignal(window=15).compute(_close_frame(close))

    assert result["volatility_15d"].iloc[:14].isna().all()
    expected = result["volatility_15d"] / close
    np.testing.assert_allclose(
        result["volatility_ratio_15d"].to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9
    )
