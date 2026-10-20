"""Portfolio-level performance metrics with a consistent annualization basis.

VectorBT's ``Portfolio.stats()`` aggregates a multi-column portfolio by taking
the *mean of each column*, which is not the portfolio's performance. It also
annualizes using the calendar length of the frequency (365 days for ``"1D"``),
which inflates Sharpe and CAGR for trading-day data.

These helpers compute metrics from a single equity curve instead, using 252
trading days per year.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def safe_float(value, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float, falling back to ``default``.

    Handles the three shapes VectorBT/pandas produce for missing metrics:
    ``None`` (key absent), ``NaN``, and ``+/-inf``.
    """
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def equity_curve(portfolio) -> pd.Series:
    """Collapse a (possibly multi-column) portfolio value frame into one curve."""
    value = portfolio.value()
    if isinstance(value, pd.DataFrame):
        return value.sum(axis=1)
    return value


def total_return(equity: pd.Series) -> float:
    """Simple return from first to last observation, as a fraction."""
    if len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    if start == 0:
        return 0.0
    return float(equity.iloc[-1]) / start - 1.0


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate on a 252-trading-day basis."""
    if len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    years = (len(equity) - 1) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def sharpe_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio on a 252-trading-day basis."""
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    std = float(excess.std(ddof=1))
    if std == 0 or math.isnan(std):
        return 0.0
    return float(excess.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio on a 252-trading-day basis.

    Downside deviation follows the standard definition - the root mean square
    of negative excess returns across *all* observations - rather than the
    standard deviation of the negative subset. The latter is undefined when
    there is only one losing period.
    """
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    downside = excess.clip(upper=0)
    downside_deviation = float(np.sqrt((downside ** 2).mean()))
    if downside_deviation == 0 or math.isnan(downside_deviation):
        return 0.0
    return float(excess.mean()) / downside_deviation * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough decline as a positive fraction (0.25 == -25%)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    worst = float(drawdown.min())
    if math.isnan(worst):
        return 0.0
    return abs(worst)


def average_drawdown(equity: pd.Series) -> float:
    """Mean drawdown across all sessions as a positive fraction."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mean = float(drawdown.mean())
    if math.isnan(mean):
        return 0.0
    return abs(mean)


def trade_win_rate(trades: pd.DataFrame, pnl_column: str = "PnL") -> float:
    """Trade-level win rate as a fraction (not a mean of per-symbol rates)."""
    if trades.empty or pnl_column not in trades.columns:
        return 0.0
    return float((trades[pnl_column] > 0).mean())


def summarize(equity: pd.Series) -> dict[str, float]:
    """Compute the standard metric bundle for a single equity curve."""
    return {
        "initial_value": safe_float(equity.iloc[0]) if len(equity) else 0.0,
        "final_value": safe_float(equity.iloc[-1]) if len(equity) else 0.0,
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "sharpe_ratio": sharpe_ratio(equity),
        "sortino_ratio": sortino_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "average_drawdown": average_drawdown(equity),
    }


def equal_weight_benchmark(close: pd.DataFrame, initial_cash: float) -> pd.Series:
    """Equal-weight buy-and-hold equity curve for the given close matrix."""
    if close.empty:
        return pd.Series(dtype=float)
    normalized = close.div(close.iloc[0])
    return normalized.mean(axis=1) * initial_cash


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Peak-to-trough decline series as negative fractions."""
    return equity / equity.cummax() - 1.0


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "average_drawdown",
    "cagr",
    "drawdown_series",
    "equal_weight_benchmark",
    "equity_curve",
    "max_drawdown",
    "safe_float",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize",
    "total_return",
    "trade_win_rate",
]
