import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbt as vbt
from quant_platform.db.connection import get_db_connection
from quant_platform.backtest.strategy import BaseStrategy

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
    """

    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads daily pricing and computed indicators from the DuckDB database.
        """
        logger.info("Loading pricing and indicators data from DuckDB...")
        
        with get_db_connection(read_only=True) as conn:
            # Check table existence
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name IN ('daily_prices', 'signals')"
            ).fetchall()
            table_names = [t[0] for t in tables]
            
            if "daily_prices" not in table_names or "signals" not in table_names:
                raise ValueError("Both 'daily_prices' and 'signals' tables must exist in DuckDB. Run ingestion and features first.")
                
            price_df = conn.execute("SELECT symbol, date, close FROM daily_prices ORDER BY date, symbol").df()
            signal_df = conn.execute("SELECT symbol, date, rsi FROM signals ORDER BY date, symbol").df()
            
        if price_df.empty or signal_df.empty:
            raise ValueError("No data found in 'daily_prices' or 'signals' tables. Run ingestion and features first.")
            
        logger.info(f"Loaded {len(price_df)} pricing records and {len(signal_df)} signal records.")
        
        # Convert date to datetime
        price_df["date"] = pd.to_datetime(price_df["date"])
        signal_df["date"] = pd.to_datetime(signal_df["date"])
        
        # Pivot DataFrames: index is Date, columns are Symbols
        close_pivot = price_df.pivot(index="date", columns="symbol", values="close")
        rsi_pivot = signal_df.pivot(index="date", columns="symbol", values="rsi")
        
        # Align indexes and columns to guarantee they match exactly
        close_pivot, rsi_pivot = close_pivot.align(rsi_pivot, join="inner", axis=0)
        close_pivot, rsi_pivot = close_pivot.align(rsi_pivot, join="inner", axis=1)
        
        # Explicitly set index as standard DatetimeIndex for VectorBT
        close_pivot.index = pd.DatetimeIndex(close_pivot.index)
        rsi_pivot.index = pd.DatetimeIndex(rsi_pivot.index)
        
        logger.info(f"Aligned historical datasets. Backtesting {len(close_pivot.columns)} symbols across {len(close_pivot)} trading sessions.")
        return close_pivot, rsi_pivot

    def run(self, init_cash: float = 10000.0, fee: float = 0.001, output_path: str = None) -> dict:
        """
        Runs the backtest.
        """
        # 1. Load aligned close and RSI matrices
        close, rsi = self.load_data()
        
        # 2. Generate signal masks
        entries, exits = self.strategy.generate_signals({"close": close, "rsi": rsi})
        
        # 3. Simulate portfolio via VectorBT
        logger.info("Simulating portfolio allocation using vectorbt...")
        # vbt.Portfolio.from_signals automatically executes trade orders on signal changes
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=fee,
            freq="1D"
        )
        
        # 4. Generate stats series
        stats = portfolio.stats()
        
        # Extract core metrics safely from stats series or direct fallbacks
        total_return = float(stats.get("Total Return [%]", 0.0)) / 100.0
        
        # Fallback to direct method for annualized return if missing/zero in stats
        raw_ann_ret = stats.get("Annualized Return [%]", None)
        if raw_ann_ret is None or float(raw_ann_ret) == 0.0:
            annualized_return = float(portfolio.annualized_return().mean())
        else:
            annualized_return = float(raw_ann_ret) / 100.0
            
        max_drawdown = float(stats.get("Max Drawdown [%]", 0.0)) / 100.0
        
        raw_sharpe = stats.get("Sharpe Ratio", None)
        if raw_sharpe is None or np.isnan(raw_sharpe) or np.isinf(raw_sharpe) or float(raw_sharpe) == 0.0:
            sharpe_ratio = float(portfolio.sharpe_ratio().mean())
        else:
            sharpe_ratio = float(raw_sharpe)
            
        win_rate = float(stats.get("Win Rate [%]", 0.0)) / 100.0
        final_value = float(stats.get("End Value", init_cash))
        
        # Clean infinite/NaN metrics
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            sharpe_ratio = 0.0
        if np.isnan(annualized_return) or np.isinf(annualized_return):
            annualized_return = 0.0

        # 5. Extract trade logs
        logger.info("Extracting portfolio trade records...")
        trades_df = portfolio.trades.records_readable
        
        trade_logs = []
        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                # Extract asset name (usually represented under Column or Col)
                symbol = row.get("Column", row.get("Col", str(row.get("symbol", "UNKNOWN"))))
                
                # Format Dates
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
                
                # Append trade log record
                trade_logs.append({
                    "symbol": str(symbol),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(row.get("Avg Entry Price", row.get("Entry Price", 0.0))),
                    "exit_price": float(row.get("Avg Exit Price", row.get("Exit Price", 0.0))),
                    "size": float(row.get("Size", 0.0)),
                    "pnl": float(row.get("PnL", 0.0)),
                    "return_pct": float(row.get("Return", 0.0)) * 100.0
                })
        
        total_trades = len(trade_logs)

        # Output Performance Tear Sheet to Console
        print("\n" + "="*50)
        print("          VECTORBT PERFORMANCE TEAR SHEET")
        print("="*50)
        print(f"Strategy Name:      {self.strategy.name}")
        print(f"Initial Capital:    ${init_cash:,.2f}")
        print(f"Final Equity:       ${final_value:,.2f}")
        print(f"Total Return:       {total_return * 100:.2f}%")
        print(f"Annualized Return:  {annualized_return * 100:.2f}%")
        print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
        print(f"Sharpe Ratio:       {sharpe_ratio:.4f}")
        print(f"Total Trades Run:   {total_trades}")
        print(f"Win Rate:           {win_rate * 100:.2f}%")
        print("="*50 + "\n")
                
        # 6. Build the export payload
        results = {
            "summary": {
                "strategy_name": self.strategy.name,
                "initial_cash": init_cash,
                "final_value": final_value,
                "total_return_pct": total_return * 100.0,
                "annualized_return_pct": annualized_return * 100.0,
                "max_drawdown_pct": max_drawdown * 100.0,
                "sharpe_ratio": sharpe_ratio,
                "total_trades": total_trades,
                "win_rate_pct": win_rate * 100.0
            },
            "trades": trade_logs
        }
        
        # 7. Write to JSON
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Successfully exported metrics and trade logs to {out_file}")
            
        return results
