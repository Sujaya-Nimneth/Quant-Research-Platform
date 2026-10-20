import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from quant_platform.config import (
    BACKTEST_RESULTS_PATH,
    DEFAULT_EXECUTION_DELAY,
    DEFAULT_FEE,
    DEFAULT_INIT_CASH,
)
from quant_platform.data_access import load_aligned_matrices
from quant_platform.metrics import (
    drawdown_series,
    equal_weight_benchmark,
    equity_curve,
    summarize,
    trade_win_rate,
)

st.set_page_config(
    page_title="Quant Research Platform - Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
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
""",
    unsafe_allow_html=True,
)

logger = logging.getLogger(__name__)

CHART_WIDTH = "stretch"


def load_backtest_results() -> dict | None:
    """Read the precomputed backtest export, if the pipeline has produced one."""
    path = Path(BACKTEST_RESULTS_PATH)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


@st.cache_data(ttl=300, show_spinner="Loading price and signal data...")
def load_market_data(indicators: tuple[str, ...]):
    """Load aligned matrices.

    Cached on the *data*, not on a connection. Holding a DuckDB connection open
    across reruns would block the ingestion and feature pipelines, which need a
    write handle on the same file.
    """
    return load_aligned_matrices(list(indicators))


def load_trades(results: dict | None) -> pd.DataFrame:
    if not results or not results.get("trades"):
        return pd.DataFrame()
    return pd.DataFrame(results["trades"])


def build_equity_chart(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_equity.index,
        y=strategy_equity.values,
        mode="lines",
        name="RSI Mean-Reversion Strategy",
        line=dict(color="#00C805", width=2.5),
        hovertemplate="Date: %{x}<br>Strategy Equity: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_equity.index,
        y=benchmark_equity.values,
        mode="lines",
        name="Equal-Weight Buy & Hold",
        line=dict(color="#8C96A6", width=1.5, dash="dash"),
        hovertemplate="Date: %{x}<br>Benchmark Equity: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,15,25,0.7)"),
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(gridcolor="#1F2937", showgrid=True, color="#A0AEC0"),
        yaxis=dict(gridcolor="#1F2937", showgrid=True, color="#A0AEC0", tickformat="$,"),
        height=500,
    )
    return fig


def build_drawdown_chart(strategy_dd: pd.Series, benchmark_dd: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_dd.index, y=strategy_dd.values * 100.0, mode="lines",
        name="Strategy", line=dict(color="#EF4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_dd.index, y=benchmark_dd.values * 100.0, mode="lines",
        name="Benchmark", line=dict(color="#8C96A6", width=1.2, dash="dash"),
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.05, bgcolor="rgba(10,15,25,0.7)"),
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(gridcolor="#1F2937", color="#A0AEC0"),
        yaxis=dict(gridcolor="#1F2937", color="#A0AEC0", ticksuffix="%"),
        height=260,
    )
    return fig


def main() -> None:
    st.title("📈 Quantitative Research Platform Core")
    st.subheader("Historical Backtest & Feature Analytics")

    try:
        close_pivot, indicators = load_market_data(("rsi",))
    except (ValueError, RuntimeError) as exc:
        st.error(f"🚨 {exc}")
        st.info(
            "Execute `python -m quant_platform.main ingest --test` followed by "
            "`python -m quant_platform.main features`."
        )
        return
    except Exception as exc:  # noqa: BLE001 - surface storage errors in the UI
        st.error(f"🚨 Failed to read the database: {exc}")
        return

    rsi_pivot = indicators["rsi"]
    results = load_backtest_results()
    trades = load_trades(results)

    if results:
        summary = results["summary"]
        equity_records = results.get("equity_curve") or []
        strategy_equity = pd.Series(
            [r["value"] for r in equity_records],
            index=pd.to_datetime([r["date"] for r in equity_records]),
            dtype=float,
        )
        total_init_cash = float(summary["initial_cash"])
        strategy_metrics = {
            "total_return": summary["total_return_pct"] / 100.0,
            "cagr": summary["annualized_return_pct"] / 100.0,
            "max_drawdown": summary["max_drawdown_pct"] / 100.0,
            "average_drawdown": summary.get("average_drawdown_pct", 0.0) / 100.0,
            "sharpe_ratio": summary["sharpe_ratio"],
            "sortino_ratio": summary.get("sortino_ratio", 0.0),
            "final_value": summary["final_value"],
        }
        win_rate = summary["win_rate_pct"] / 100.0
        total_trades = summary["total_trades"]
        execution_delay = summary.get("execution_delay_bars", DEFAULT_EXECUTION_DELAY)
        fee = summary.get("fee", DEFAULT_FEE)
    else:
        st.warning(
            "No backtest export found. Run `python -m quant_platform.main backtest` "
            "to generate one. Showing price data only."
        )
        return

    # Benchmark is equal-weight buy & hold over the same universe and capital.
    benchmark_equity = equal_weight_benchmark(close_pivot, total_init_cash)
    benchmark_metrics = summarize(benchmark_equity)

    strategy_equity = strategy_equity.reindex(benchmark_equity.index).ffill()

    # --- Sidebar KPIs ---
    st.sidebar.header("📊 Strategy Performance")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Capital ${total_init_cash:,.0f} · fee {fee * 100:.2f}% · {execution_delay}-bar execution delay"
    )

    strat_final = float(strategy_equity.iloc[-1])
    st.sidebar.metric(
        label="Strategy Final Equity",
        value=f"${strat_final:,.2f}",
        delta=f"{strategy_metrics['total_return'] * 100:+.2f}%",
    )
    bench_final = float(benchmark_equity.iloc[-1])
    st.sidebar.metric(
        label="Benchmark Final Equity",
        value=f"${bench_final:,.2f}",
        delta=f"{benchmark_metrics['total_return'] * 100:+.2f}%",
        delta_color="off",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Key Ratios")
    col_kpi1, col_kpi2 = st.sidebar.columns(2)
    with col_kpi1:
        st.metric("Sharpe", f"{strategy_metrics['sharpe_ratio']:.3f}")
        st.metric("Win Rate", f"{win_rate * 100:.1f}%")
    with col_kpi2:
        st.metric("Max Drawdown", f"{strategy_metrics['max_drawdown'] * 100:.1f}%")
        st.metric("CAGR", f"{strategy_metrics['cagr'] * 100:.1f}%")
    st.sidebar.metric("Total Executed Trades", f"{total_trades}")

    tab1, tab2, tab3 = st.tabs([
        "📈 Portfolio Equity Curves",
        "📑 Trades Auditor",
        "🔍 Single Ticker Inspector",
    ])

    with tab1:
        st.write("#### Strategy vs. Equal-Weight Buy & Hold")
        st.plotly_chart(
            build_equity_chart(strategy_equity, benchmark_equity),
            width=CHART_WIDTH,
        )

        st.write("#### Performance Summary Comparison")
        metrics_comp = pd.DataFrame({
            "Metric": [
                "Initial Capital", "Ending Equity", "Total Return", "CAGR (252d)",
                "Sharpe (252d)", "Max Drawdown", "Average Drawdown",
            ],
            "RSI Strategy": [
                f"${total_init_cash:,.2f}",
                f"${strat_final:,.2f}",
                f"{strategy_metrics['total_return'] * 100:.2f}%",
                f"{strategy_metrics['cagr'] * 100:.2f}%",
                f"{strategy_metrics['sharpe_ratio']:.3f}",
                f"{strategy_metrics['max_drawdown'] * 100:.2f}%",
                f"{strategy_metrics['average_drawdown'] * 100:.2f}%",
            ],
            "Equal-Weight Buy & Hold": [
                f"${total_init_cash:,.2f}",
                f"${bench_final:,.2f}",
                f"{benchmark_metrics['total_return'] * 100:.2f}%",
                f"{benchmark_metrics['cagr'] * 100:.2f}%",
                f"{benchmark_metrics['sharpe_ratio']:.3f}",
                f"{benchmark_metrics['max_drawdown'] * 100:.2f}%",
                f"{benchmark_metrics['average_drawdown'] * 100:.2f}%",
            ],
        })
        st.table(metrics_comp)

        st.write("#### Drawdown")
        st.plotly_chart(
            build_drawdown_chart(
                drawdown_series(strategy_equity),
                drawdown_series(benchmark_equity),
            ),
            width=CHART_WIDTH,
        )

    with tab2:
        st.write("#### Complete Backtesting Trades Audit Log")
        if trades.empty:
            st.warning("No trades were executed during this period.")
        else:
            tickers_in_trades = sorted(trades["symbol"].unique())
            default_tickers = tickers_in_trades[:5] if len(tickers_in_trades) > 5 else tickers_in_trades
            selected_tickers = st.multiselect(
                "Filter by Ticker Symbol(s)", tickers_in_trades, default=default_tickers
            )
            filtered = trades[trades["symbol"].isin(selected_tickers)].copy()

            st.dataframe(
                filtered.style.format({
                    "entry_price": "${:,.2f}",
                    "exit_price": "${:,.2f}",
                    "size": "{:,.2f}",
                    "pnl": "${:,.2f}",
                    "return_pct": "{:.2f}%",
                }),
                width=CHART_WIDTH,
            )

            col_tr1, col_tr2, col_tr3 = st.columns(3)
            col_tr1.metric("Largest Win ($)", f"${trades['pnl'].max():,.2f}")
            col_tr2.metric("Largest Loss ($)", f"${trades['pnl'].min():,.2f}")
            col_tr3.metric("Average Profit per Trade ($)", f"${trades['pnl'].mean():,.2f}")

            st.caption(f"Trade-level win rate: {trade_win_rate(trades, 'pnl') * 100:.2f}%")

    with tab3:
        st.write("#### Single Asset Pricing & Indicator Chart Auditor")
        selected_ticker = st.selectbox("Select Asset to Audit", sorted(close_pivot.columns))

        asset_close = close_pivot[selected_ticker]
        asset_rsi = rsi_pivot[selected_ticker]
        asset_entries = (asset_rsi < 30).fillna(False)
        asset_exits = (asset_rsi > 70).fillna(False)

        fig_ind = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4]
        )
        fig_ind.add_trace(go.Scatter(
            x=asset_close.index, y=asset_close.values, mode="lines", name="Close Price",
            line=dict(color="#3B82F6", width=2),
            hovertemplate="Price: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

        fig_ind.add_trace(go.Scatter(
            x=asset_close.index[asset_entries], y=asset_close.values[asset_entries],
            mode="markers", name="Buy (RSI < 30)",
            marker=dict(symbol="triangle-up", color="#10B981", size=11, line=dict(color="white", width=1)),
            hovertemplate="Buy Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

        fig_ind.add_trace(go.Scatter(
            x=asset_close.index[asset_exits], y=asset_close.values[asset_exits],
            mode="markers", name="Sell (RSI > 70)",
            marker=dict(symbol="triangle-down", color="#EF4444", size=11, line=dict(color="white", width=1)),
            hovertemplate="Sell Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

        fig_ind.add_trace(go.Scatter(
            x=asset_rsi.index, y=asset_rsi.values, mode="lines", name="RSI (14)",
            line=dict(color="#F59E0B", width=1.5),
            hovertemplate="RSI Value: %{y:.2f}<extra></extra>",
        ), row=2, col=1)

        fig_ind.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=2, col=1,
                          annotation_text="Overbought (70)", annotation_position="top left")
        fig_ind.add_hline(y=30, line_dash="dash", line_color="#10B981", row=2, col=1,
                          annotation_text="Oversold (30)", annotation_position="bottom left")

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
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,15,25,0.7)"),
        )
        st.plotly_chart(fig_ind, width=CHART_WIDTH)


if __name__ == "__main__":
    main()
