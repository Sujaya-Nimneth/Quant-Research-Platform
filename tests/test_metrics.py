"""Portfolio metric maths, including the annualization basis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_platform.metrics import (
    TRADING_DAYS_PER_YEAR,
    average_drawdown,
    cagr,
    drawdown_series,
    equal_weight_benchmark,
    equity_curve,
    max_drawdown,
    safe_float,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    trade_win_rate,
)


class _StubPortfolio:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


def test_safe_float_handles_all_missing_shapes():
    assert safe_float(None) == 0.0
    assert safe_float(float("nan")) == 0.0
    assert safe_float(float("inf")) == 0.0
    assert safe_float("-inf", default=-1.0) == -1.0
    assert safe_float("not a number") == 0.0
    assert safe_float(3.5) == 3.5


def test_equity_curve_sums_columns():
    frame = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 90.0]})
    curve = equity_curve(_StubPortfolio(frame))
    assert curve.tolist() == [200.0, 200.0]


def test_equity_curve_passes_through_series():
    series = pd.Series([100.0, 120.0])
    assert equity_curve(_StubPortfolio(series)).equals(series)


def test_total_return():
    curve = pd.Series([100.0, 150.0])
    assert total_return(curve) == pytest.approx(0.5)


def test_total_return_is_zero_for_degenerate_input():
    assert total_return(pd.Series([100.0])) == 0.0
    assert total_return(pd.Series([0.0, 10.0])) == 0.0


def test_cagr_uses_252_day_basis():
    # Doubling over exactly one trading year must equal 100% CAGR.
    n = TRADING_DAYS_PER_YEAR + 1
    curve = pd.Series(np.linspace(100.0, 200.0, n))
    assert cagr(curve) == pytest.approx(1.0, rel=1e-6)


def test_cagr_uses_elapsed_time_when_indexed():
    # Doubling over one calendar year of sessions, whatever their count.
    index = pd.date_range("2024-01-01", "2024-12-31", freq="25h")
    curve = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index)
    assert cagr(curve) == pytest.approx(1.0, rel=0.02)


def test_sharpe_annualizes_on_252_days():
    rng = np.random.default_rng(3)
    curve = pd.Series(100 * (1 + rng.normal(0.001, 0.01, 500)).cumprod())

    returns = curve.pct_change().dropna()
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert sharpe_ratio(curve) == pytest.approx(expected, rel=1e-9)

    # The same data annualized over 365 calendar days (the old behaviour)
    # must be materially higher, confirming the basis actually changed.
    calendar_basis = returns.mean() / returns.std(ddof=1) * math.sqrt(365)
    assert calendar_basis > expected


def test_sharpe_is_zero_for_flat_series():
    assert sharpe_ratio(pd.Series([100.0] * 50)) == 0.0


def test_sharpe_is_infinite_for_monotonic_winners():
    # Constant-ratio growth: every return is identical, so std == 0 with a
    # positive drift -> inf, not a masked 0.0.
    curve = pd.Series(100.0 * 1.01 ** np.arange(20))
    assert sharpe_ratio(curve) == math.inf


def test_ratio_is_negative_infinite_for_monotonic_losers():
    curve = pd.Series(100.0 * 0.99 ** np.arange(20))
    assert sharpe_ratio(curve) == -math.inf
    # Every return is downside, so Sortino stays finite at -sqrt(252).
    assert sortino_ratio(curve) == pytest.approx(-math.sqrt(TRADING_DAYS_PER_YEAR))


def test_sharpe_strips_inf_spikes_from_zero_crossings():
    # A curve dipping through zero produces +/-inf pct-change spikes that
    # would poison the mean and std; they must be dropped instead.
    curve = pd.Series([100.0, 50.0, 0.0, 25.0, 60.0])
    result = sharpe_ratio(curve)
    assert math.isfinite(result)


def test_sortino_ignores_upside_volatility():
    curve = pd.Series([100.0, 101.0, 100.5, 102.0, 103.0])
    assert sortino_ratio(curve) > 0


def test_sortino_is_infinite_without_downside():
    # Only upside returns -> zero downside deviation -> inf, not 0.0.
    assert sortino_ratio(pd.Series([100.0, 101.0, 102.0, 103.0])) == math.inf


def test_max_drawdown_is_positive_fraction():
    curve = pd.Series([100.0, 120.0, 90.0, 130.0])
    assert max_drawdown(curve) == pytest.approx(0.25)


def test_average_drawdown_is_positive():
    curve = pd.Series([100.0, 80.0, 100.0])
    assert average_drawdown(curve) > 0
    assert (drawdown_series(curve) <= 0).all()


def test_win_rate_is_trade_level_not_symbol_mean():
    trades = pd.DataFrame({"pnl": [10.0, -5.0, 20.0, 30.0]})
    assert trade_win_rate(trades, "pnl") == pytest.approx(0.75)
    assert trade_win_rate(pd.DataFrame(), "pnl") == 0.0


def test_equal_weight_benchmark_starts_at_initial_cash():
    close = pd.DataFrame({"A": [10.0, 11.0], "B": [20.0, 22.0]})
    curve = equal_weight_benchmark(close, 10_000.0)
    assert curve.iloc[0] == pytest.approx(10_000.0)
    assert curve.iloc[-1] == pytest.approx(11_000.0)


def test_equal_weight_benchmark_normalizes_late_listed_symbol_from_own_start():
    # B only has prices from the second row; it must still contribute from
    # that date instead of being dropped as an all-NaN column.
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [np.nan, 20.0, 24.0]})
    curve = equal_weight_benchmark(close, 10_000.0)
    assert curve.iloc[0] == pytest.approx(10_000.0)
    # Last row: A +20% (10 -> 12), B +20% (20 -> 24) -> mean +20%.
    assert curve.iloc[-1] == pytest.approx(12_000.0)


def test_equal_weight_benchmark_empty_frame():
    empty = pd.DataFrame()
    assert equal_weight_benchmark(empty, 10_000.0).empty
