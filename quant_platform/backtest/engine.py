import json
import logging
from pathlib import Path

import pandas as pd
import vectorbt as vbt

from quant_platform.backtest.strategy import BaseStrategy
from quant_platform.config import DEFAULT_EXECUTION_DELAY
from quant_platform.data_access import load_aligned_matrices
from quant_platform.metrics import (
    drawdown_series,
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
        init_cash: float = 10000.0,
        fee: float = 0.001,
        output_path: str = None,
        slippage: float = 0.0,
    ) -> dict:
        """
        Runs the backtest.

        ``init_cash`` is the total portfolio capital, shared across all symbols.
        """
        # 1. Load aligned close and indicator matrices
        close, indicators = self.load_data()

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
        total_trades = len(trades_df)
        win_rate = trade_win_rate(trades_df)

        trade_logs = self._extract_trade_logs(trades_df)

        self._print_tear_sheet(init_cash, metrics, total_trades, win_rate)

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
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "total_trades": total_trades,
                "win_rate_pct": win_rate * 100.0,
            },
            "equity_curve": [
                {"date": index.strftime("%Y-%m-%d"), "value": safe_float(value)} for index, value in equity.items()
            ],
            "trades": trade_logs,
        }

        # 7. Write to JSON
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Successfully exported metrics and trade logs to {out_file}")

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

            if hasattr(entry_date, "strftime"):
                entry_date = entry_date.strftime("%Y-%m-%d")
            else:
                entry_date = str(entry_date).split(" ")[0]

            if hasattr(exit_date, "strftime"):
                exit_date = exit_date.strftime("%Y-%m-%d")
            else:
                exit_date = str(exit_date).split(" ")[0]

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

    def _print_tear_sheet(self, init_cash: float, metrics: dict, total_trades: int, win_rate: float) -> None:
        print("\n" + "=" * 50)
        print("          VECTORBT PERFORMANCE TEAR SHEET")
        print("=" * 50)
        print(f"Strategy Name:      {self.strategy.name}")
        print(f"Execution Delay:    {self.execution_delay} bar(s)")
        print(f"Initial Capital:    ${init_cash:,.2f}")
        print(f"Final Equity:       ${metrics['final_value']:,.2f}")
        print(f"Total Return:       {metrics['total_return'] * 100:.2f}%")
        print(f"Annualized Return:  {metrics['cagr'] * 100:.2f}%")
        print(f"Max Drawdown:       {metrics['max_drawdown'] * 100:.2f}%")
        print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio:      {metrics['sortino_ratio']:.4f}")
        print(f"Total Trades Run:   {total_trades}")
        print(f"Win Rate:           {win_rate * 100:.2f}%")
        print("=" * 50 + "\n")


__all__ = ["BacktestEngine", "drawdown_series"]
