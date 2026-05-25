import logging
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt

# Suppress Streamlit warnings
st.set_page_config(
    page_title="Quant Research Platform - Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS Inject)
st.markdown("""
<style>
    /* Premium aesthetics */
    .stApp {
        background-color: #0E1117;
        color: #E0E2E6;
    }
    .stMetric {
        background-color: #1A1D24 !important;
        border: 1px solid #2B303B !important;
        border-radius: 8px !important;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
    }
</style>
""", unsafe_allowed_name=True, unsafe_allowed_html=True)

logger = logging.getLogger(__name__)

# Cache database connection
@st.cache_resource
def get_duckdb_conn():
    # Return a read-only DuckDB connection
    return duckdb.connect("data/quant_platform.db", read_only=True)

def load_backtest_json():
    json_path = Path("data/backtest_results.json")
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        return json.load(f)

# Main App Structure
def main():
    st.title("📈 Quantitative Research Platform Core")
    st.subheader("High-Performance Historical Backtest & Feature Analytics")
    
    # 1. Check Data Sourcing
    conn = get_duckdb_conn()
    json_data = load_backtest_json()
    
    # Verify tables exist
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    
    if "daily_prices" not in table_names or "signals" not in table_names:
        st.error("🚨 Database tables 'daily_prices' or 'signals' not found. Please run the ingestion and feature pipeline first!")
        st.info("Execute: `python -m quant_platform.main ingest --test` followed by `python -m quant_platform.main features`")
        return
        
    # 2. Programmatic VectorBT Simulation for Aligned Equity Calculations
    price_df = conn.execute("SELECT symbol, date, close FROM daily_prices ORDER BY date, symbol").df()
    signal_df = conn.execute("SELECT symbol, date, rsi FROM signals ORDER BY date, symbol").df()
    
    # Convert date and pivot
    price_df["date"] = pd.to_datetime(price_df["date"])
    signal_df["date"] = pd.to_datetime(signal_df["date"])
    close_pivot = price_df.pivot(index="date", columns="symbol", values="close")
    rsi_pivot = signal_df.pivot(index="date", columns="symbol", values="rsi")
    
    # Align matrices
    close_pivot, rsi_pivot = close_pivot.align(rsi_pivot, join="inner", axis=0)
    close_pivot, rsi_pivot = close_pivot.align(rsi_pivot, join="inner", axis=1)
    close_pivot.index = pd.DatetimeIndex(close_pivot.index)
    rsi_pivot.index = pd.DatetimeIndex(rsi_pivot.index)
    
    num_symbols = len(close_pivot.columns)
    init_cash_per_symbol = 10000.0
    total_init_cash = init_cash_per_symbol * num_symbols
    
    # Simulate RSI strategy matching backtest parameters
    entries = rsi_pivot < 30
    exits = rsi_pivot > 70
    
    portfolio = vbt.Portfolio.from_signals(
        close=close_pivot,
        entries=entries,
        exits=exits,
        init_cash=init_cash_per_symbol,
        fees=0.001,
        freq="1D"
    )
    
    # Calculate curves
    strategy_equity = portfolio.value().sum(axis=1)
    
    # Normalized buy-and-hold benchmark curve (apples-to-apples)
    normalized_close = close_pivot.div(close_pivot.iloc[0])
    benchmark_equity = normalized_close.sum(axis=1) * init_cash_per_symbol
    
    # Retrieve metrics
    total_trades_run = len(portfolio.trades.records_readable)
    avg_win_rate = portfolio.trades.win_rate().mean() * 100.0
    avg_ann_ret = portfolio.annualized_return().mean() * 100.0
    avg_max_dd = portfolio.max_drawdown().mean() * 100.0
    avg_sharpe = portfolio.sharpe_ratio().mean()
    
    # Clean Sharpe if NaN/inf
    import numpy as np
    if np.isnan(avg_sharpe) or np.isinf(avg_sharpe):
        avg_sharpe = 0.0

    # 3. Sidebar KPI Summary Cards
    st.sidebar.header("📊 Strategy Performance")
    st.sidebar.markdown("---")
    
    st.sidebar.metric(
        label="Total Starting Capital",
        value=f"${total_init_cash:,.2f}"
    )
    
    strat_final = strategy_equity.iloc[-1]
    strat_return = ((strat_final - total_init_cash) / total_init_cash) * 100.0
    st.sidebar.metric(
        label="Strategy Final Equity",
        value=f"${strat_final:,.2f}",
        delta=f"{strat_return:+.2f}%"
    )
    
    bench_final = benchmark_equity.iloc[-1]
    bench_return = ((bench_final - total_init_cash) / total_init_cash) * 100.0
    st.sidebar.metric(
        label="Benchmark Final Equity",
        value=f"${bench_final:,.2f}",
        delta=f"{bench_return:+.2f}%",
        delta_color="off"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Key Ratios")
    
    col_kpi1, col_kpi2 = st.sidebar.columns(2)
    with col_kpi1:
        st.metric("Sharpe Ratio", f"{avg_sharpe:.4f}")
        st.metric("Win Rate", f"{avg_win_rate:.1f}%")
    with col_kpi2:
        st.metric("Max Drawdown", f"{avg_max_dd:.1f}%")
        st.metric("Annualized (CAGR)", f"{avg_ann_ret:.1f}%")
        
    st.sidebar.metric("Total Executed Trades", f"{total_trades_run}")

    # 4. Tabbed Main Dashboard Panels
    tab1, tab2, tab3 = st.tabs([
        "📈 Portfolio Equity Curves", 
        "📑 Trades Auditor", 
        "🔍 Single Ticker Inspector"
    ])
    
    # --- TAB 1: Equity Curve Comparison ---
    with tab1:
        st.write("#### Strategy vs. Buy-and-Hold S&P 500 Cumulative Return")
        
        # Plotly Line Chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=strategy_equity.index,
            y=strategy_equity.values,
            mode="lines",
            name="RSI Mean-Reversion Strategy",
            line=dict(color="#00C805", width=2.5),
            hovertemplate="Date: %{x}<br>Strategy Equity: $%{y:,.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity.values,
            mode="lines",
            name="S&P 500 Buy-and-Hold Benchmark",
            line=dict(color="#8C96A6", width=1.5, dash="dash"),
            hovertemplate="Date: %{x}<br>Benchmark Equity: $%{y:,.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,15,25,0.7)"),
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(gridcolor="#1F2937", showgrid=True, color="#A0AEC0"),
            yaxis=dict(gridcolor="#1F2937", showgrid=True, color="#A0AEC0", tickformat="$,"),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Diagnostics Table
        st.write("#### Performance Summary Comparison")
        metrics_comp = pd.DataFrame({
            "Metric": ["Initial Capital", "Ending Equity", "Total Return", "Avg Drawdown", "Max Drawdown"],
            "RSI Strategy": [
                f"${total_init_cash:,.2f}",
                f"${strat_final:,.2f}",
                f"{strat_return:.2f}%",
                f"{portfolio.drawdown().mean().mean() * -100:.2f}%",
                f"{avg_max_dd:.2f}%"
            ],
            "Buy & Hold S&P 500": [
                f"${total_init_cash:,.2f}",
                f"${bench_final:,.2f}",
                f"{bench_return:.2f}%",
                f"{normalized_close.pct_change().std().mean() * 100:.2f}% (Daily Vol)",  # proxy
                f"{(normalized_close.div(normalized_close.cummax()) - 1).min().mean() * 100:.2f}%"
            ]
        })
        st.table(metrics_comp)

    # --- TAB 2: Interactive Trades Logger ---
    with tab2:
        st.write("#### Complete Backtesting Trades Audit Log")
        
        # Load Trades
        trades_df = portfolio.trades.records_readable
        
        if trades_df.empty:
            st.warning("No trades were executed during this period.")
        else:
            # Clean Ticker/Column Representation
            trades_df["Column"] = trades_df["Column"].astype(str)
            trades_df = trades_df.rename(columns={"Column": "Ticker"})
            
            # Interactive Filter by Symbol
            tickers_in_trades = sorted(trades_df["Ticker"].unique())
            selected_tickers = st.multiselect("Filter by Ticker Symbol(s)", tickers_in_trades, default=tickers_in_trades[:5] if len(tickers_in_trades) > 5 else tickers_in_trades)
            
            filtered_trades = trades_df[trades_df["Ticker"].isin(selected_tickers)].copy()
            
            # Formatting floats
            format_cols = {
                "Size": "{:,.2f}",
                "Avg Entry Price": "${:,.2f}",
                "Avg Exit Price": "${:,.2f}",
                "Entry Fees": "${:,.2f}",
                "Exit Fees": "${:,.2f}",
                "PnL": "${:,.2f}",
                "Return": "{:.2f}%"
            }
            
            # Render
            rendered_df = filtered_trades.copy()
            # Map Return to percentage directly
            rendered_df["Return"] = rendered_df["Return"] * 100.0
            
            st.dataframe(
                rendered_df.style.format(format_cols),
                use_container_width=True
            )
            
            # Display Trade Stats
            col_tr1, col_tr2, col_tr3 = st.columns(3)
            with col_tr1:
                st.metric("Largest Win ($)", f"${trades_df['PnL'].max():,.2f}")
            with col_tr2:
                st.metric("Largest Loss ($)", f"${trades_df['PnL'].min():,.2f}")
            with col_tr3:
                st.metric("Average Profit per Trade ($)", f"${trades_df['PnL'].mean():,.2f}")

    # --- TAB 3: Indicator and Signal Inspector ---
    with tab3:
        st.write("#### Single Asset Pricing & Indicator Chart Auditor")
        
        selected_ticker = st.selectbox("Select Asset to Audit", sorted(close_pivot.columns))
        
        # Slices
        asset_close = close_pivot[selected_ticker]
        asset_rsi = rsi_pivot[selected_ticker]
        asset_entries = entries[selected_ticker]
        asset_exits = exits[selected_ticker]
        
        # Subplots: Price (Top) and RSI (Bottom)
        fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
        
        # Price Line
        fig_ind.add_trace(go.Scatter(
            x=asset_close.index,
            y=asset_close.values,
            mode="lines",
            name="Close Price",
            line=dict(color="#3B82F6", width=2),
            hovertemplate="Price: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
        
        # Buy Markers
        buy_dates = asset_close.index[asset_entries]
        buy_prices = asset_close.values[asset_entries]
        fig_ind.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode="markers",
            name="Buy (RSI < 30)",
            marker=dict(symbol="triangle-up", color="#10B981", size=11, line=dict(color="white", width=1)),
            hovertemplate="Buy Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
        
        # Sell Markers
        sell_dates = asset_close.index[asset_exits]
        sell_prices = asset_close.values[asset_exits]
        fig_ind.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode="markers",
            name="Sell (RSI > 70)",
            marker=dict(symbol="triangle-down", color="#EF4444", size=11, line=dict(color="white", width=1)),
            hovertemplate="Sell Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
        
        # RSI Line
        fig_ind.add_trace(go.Scatter(
            x=asset_rsi.index,
            y=asset_rsi.values,
            mode="lines",
            name="RSI (14)",
            line=dict(color="#F59E0B", width=1.5),
            hovertemplate="RSI Value: %{y:.2f}<extra></extra>"
        ), row=2, col=1)
        
        # Add Horizontal bands for RSI threshold bounds
        fig_ind.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=2, col=1, annotation_text="Overbought (70)", annotation_position="top left")
        fig_ind.add_hline(y=30, line_dash="dash", line_color="#10B981", row=2, col=1, annotation_text="Oversold (30)", annotation_position="bottom left")
        
        fig_ind.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            xaxis2=dict(gridcolor="#1F2937", color="#A0AEC0", showgrid=True),
            xaxis=dict(gridcolor="#1F2937", color="#A0AEC0", showgrid=True),
            yaxis=dict(gridcolor="#1F2937", color="#A0AEC0", tickformat="$", showgrid=True, title="Asset Price"),
            yaxis2=dict(gridcolor="#1F2937", color="#A0AEC0", range=[0, 100], showgrid=True, title="RSI"),
            height=600,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,15,25,0.7)")
        )
        
        st.plotly_chart(fig_ind, use_container_width=True)

if __name__ == "__main__":
    main()
