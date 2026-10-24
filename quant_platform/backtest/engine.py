import json
import logging
import math
from pathlib import Path

import pandas as pd
import vectorbt as vbt

from quant_platform import config
from quant_platform.backtest.strategy import BaseStrategy
from quant_platform.config import DEFAULT_EXECUTION_DELAY
from quant_platform.data_access import load_aligned_matrices
from quant_platform.metrics import (
    equity_curve,
    safe_float,
    summarize,
    trade_win_rate,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine utilizing the vectorbt library.
    It:
    1. Loads prices and signals from DuckDB.
    2. Runs a designated signal-based strategy.
    3. Simulates the portfolio over time with trading fees.
    4. Outputs a performance tear sheet.
    5. Exports detailed trade logs and performance metrics to a JSON file.

    All reported metrics are computed from the *aggregated* portfolio equity
    curve, not from a mean of per-symbol statistics, and are annualized on a
    252-trading-day basis.
    """

    def __init__(self, strategy: BaseStrategy, execution_delay: int = DEFAULT_EXECUTION_DELAY):
        if execution_delay < 1:
            raise ValueError(
                "execution_delay must be >= 1 so signals computed from a bar's close "
                "cannot be filled at that same close (look-ahead bias)."
            )
        self.strategy = strategy
        self.execution_delay = execution_delay

    def load_data(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Loads daily pricing and the indicator columns required by the strategy.
        """
        logger.info("Loading pricing and indicators data from DuckDB...")
        return load_aligned_matrices(self.strategy.required_indicators)

    def run(
        self,
        init_cash: float = config.DEFAULT_INIT_CASH,
        fee: float = config.DEFAULT_FEE,
        output_path: str | None = None,
        slippage: float = 0.0,
    ) -> dict:
        """
        Runs the backtest.

        ``init_cash`` is the total portfolio capital, shared across all symbols.
        """
        # 1. Load aligned close and indicator matrices
        close, indicators = self.load_data()

        # Drop symbols with no prices at all: they can never trade, and an
        # equal cash slice reserved for a dead column just drags the metrics.
        tradable = [col for col in close.columns if close[col].notna().any()]
        if not tradable:
            raise ValueError("No symbol has any price data; nothing to backtest.")
        if len(tradable) < len(close.columns):
            dead = [str(col) for col in close.columns if col not in tradable]
            logger.warning("Excluding all-NaN symbol(s) from the backtest: %s", ", ".join(dead))
            close = close[tradable]
            indicators = {name: matrix[tradable] for name, matrix in indicators.items()}

        # 2. Generate signal masks on the strategy's own timeframe
        data_dict = {"close": close, **indicators}
        entries, exits = self.strategy.generate_signals(data_dict)

        # 3. Execute on the next bar. A signal derived from bar t's close is only
        #    knowable after that close, so it cannot be filled at the same price.
        entries = entries.shift(self.execution_delay).fillna(False).astype(bool)
        exits = exits.shift(self.execution_delay).fillna(False).astype(bool)

        # 4. Simulate the portfolio via VectorBT.
        #    ``init_cash`` is total capital: it is split evenly across symbols so
        #    every name can hold a position and diversification is preserved.
        #    The split is static - idle cash in one symbol's book cannot fund
        #    another's entry - which makes the simulation conservative.
        cash_per_symbol = init_cash / len(close.columns)
        logger.info(
            "Simulating portfolio allocation using vectorbt (delay=%d bar(s), $%.2f per symbol)...",
            self.execution_delay,
            cash_per_symbol,
        )
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=cash_per_symbol,
            fees=fee,
            slippage=slippage,
            freq="1D",
        )

        # 5. Portfolio-level metrics from the aggregated equity curve
        equity = equity_curve(portfolio)
        metrics = summarize(equity)

        trades_df = portfolio.trades.records_readable
        # Open trades (still held at the last bar) have no realized PnL and are
        # excluded from the win rate instead of counting as losses.
        closed_trades = (
            trades_df[trades_df["Exit Timestamp"].notna()] if "Exit Timestamp" in trades_df.columns else trades_df
        )
        total_trades = len(trades_df)
        win_rate = trade_win_rate(closed_trades)

        trade_logs = self._extract_trade_logs(trades_df)

        self._log_tear_sheet(init_cash, metrics, total_trades, win_rate)

        # 6. Build the export payload
        results = {
            "summary": {
                "strategy_name": self.strategy.name,
                "fee": fee,
                "slippage": slippage,
                "execution_delay_bars": self.execution_delay,
                "annualization_basis_days": 252,
                "initial_cash": init_cash,
                "final_value": metrics["final_value"],
                "total_return_pct": metrics["total_return"] * 100.0,
                "annualized_return_pct": metrics["cagr"] * 100.0,
                "max_drawdown_pct": metrics["max_drawdown"] * 100.0,
                "average_drawdown_pct": metrics["average_drawdown"] * 100.0,
                "sharpe_ratio": _json_number(metrics["sharpe_ratio"]),
                "sortino_ratio": _json_number(metrics["sortino_ratio"]),
                "total_trades": total_trades,
                "win_rate_pct": win_rate * 100.0,
            },
            "equity_curve": [
                {"date": index.strftime("%Y-%m-%d"), "value": safe_float(value)} for index, value in equity.items()
            ],
            "trades": trade_logs,
        }

        # 7. Write to JSON atomically: a crash mid-write must not leave a
        #    truncated file that the dashboard then fails to parse.
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
            with open(tmp_file, "w") as f:
                json.dump(results, f, indent=4, allow_nan=False)
                f.write("\n")
            tmp_file.replace(out_file)
            logger.info("Successfully exported metrics and trade logs to %s", out_file)

        return results

    @staticmethod
    def _extract_trade_logs(trades_df: pd.DataFrame) -> list[dict]:
        """Normalize VectorBT's readable trade records into plain dictionaries."""
        trade_logs = []
        if trades_df.empty:
            return trade_logs

        for _, row in trades_df.iterrows():
            symbol = row.get("Column", row.get("Col", str(row.get("symbol", "UNKNOWN"))))

            entry_date = row.get("Entry Timestamp", row.get("Entry Date", ""))
            exit_date = row.get("Exit Timestamp", row.get("Exit Date", ""))

            entry_date = _format_trade_date(entry_date)
            # Still-held trades export as null instead of the string "NaT".
            exit_date = None if pd.isna(exit_date) else _format_trade_date(exit_date)

            trade_logs.append(
                {
                    "symbol": str(symbol),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": safe_float(row.get("Avg Entry Price", row.get("Entry Price"))),
                    "exit_price": safe_float(row.get("Avg Exit Price", row.get("Exit Price"))),
                    "size": safe_float(row.get("Size")),
                    "pnl": safe_float(row.get("PnL")),
                    "return_pct": safe_float(row.get("Return")) * 100.0,
                }
            )

        return trade_logs

    def _log_tear_sheet(self, init_cash: float, metrics: dict, total_trades: int, win_rate: float) -> None:
        logger.info(
            "Tear sheet | strategy=%s delay=%d bars | initial $%.2f -> final $%.2f | "
            "return %.2f%% (CAGR %.2f%%) | max DD %.2f%% | Sharpe %s Sortino %s | "
            "%d trades, %.2f%% win rate",
            self.strategy.name,
            self.execution_delay,
            init_cash,
            metrics["final_value"],
            metrics["total_return"] * 100.0,
            metrics["cagr"] * 100.0,
            metrics["max_drawdown"] * 100.0,
            metrics["sharpe_ratio"],
            metrics["sortino_ratio"],
            total_trades,
            win_rate * 100.0,
        )


def _format_trade_date(value) -> str | None:
    """Render a trade timestamp as YYYY-MM-DD; NaT/empty become None."""
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).split(" ")[0]


def _json_number(value: float) -> float | None:
    """Map non-finite metrics (e.g. an infinite Sharpe) to null for strict JSON."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


__all__ = ["BacktestEngine"]
