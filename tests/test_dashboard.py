"""Dashboard smoke tests using Streamlit's AppTest harness."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "quant_platform" / "app.py")


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    """st.cache_data persists across AppTest runs, which would leak fixtures."""
    streamlit.cache_data.clear()
    yield
    streamlit.cache_data.clear()


@pytest.fixture(scope="module")
def dashboard_env(tmp_path_factory):
    """Seed a DB and a backtest export once per module.

    Running the full feature + backtest pipeline is expensive; every dashboard
    test in this module reads the same seeded state.
    """
    import quant_platform.config as cfg
    from quant_platform.backtest.engine import BacktestEngine
    from quant_platform.backtest.strategy import RSIStrategy
    from quant_platform.db.schema import init_db
    from quant_platform.features.factory import SignalFactory

    data_dir = tmp_path_factory.mktemp("dashboard")
    db_path = data_dir / "test.db"
    results_path = data_dir / "backtest_results.json"

    conn = duckdb.connect(str(db_path))
    init_db(conn)
    rows = []
    price = 100.0
    import datetime as dt
    import math

    for symbol, offset in (("AAA", 0.0), ("BBB", 1.5)):
        price = 100.0
        for i in range(120):
            price *= 1.0 + 0.004 * math.sin(i / 3.0) + 0.001 * math.cos(i / 7.0)
            rows.append(
                (
                    symbol,
                    dt.date(2024, 1, 1) + dt.timedelta(days=i),
                    price * 0.995,
                    price * 1.01,
                    price * 0.99,
                    price + offset,
                    price + offset,
                    1_000_000 + i,
                )
            )
    conn.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)", rows)
    conn.close()

    old_db, old_results = cfg.DB_PATH, cfg.BACKTEST_RESULTS_PATH
    cfg.DB_PATH = db_path
    cfg.BACKTEST_RESULTS_PATH = results_path
    try:
        SignalFactory().run_pipeline()
        results = BacktestEngine(RSIStrategy()).run(init_cash=10_000.0)
        results_path.write_text(json.dumps(results))
        yield db_path
    finally:
        cfg.DB_PATH = old_db
        cfg.BACKTEST_RESULTS_PATH = old_results
        streamlit.cache_data.clear()


def test_dashboard_renders_without_exception(dashboard_env):
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()

    assert not at.exception, [e.message for e in at.exception]
    assert not at.error
    assert len(at.tabs) == 3


def test_dashboard_shows_strategy_metrics(dashboard_env):
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()

    labels = [m.label for m in at.metric]
    assert "Strategy Final Equity" in labels
    assert "Sharpe" in labels
    assert "Max Drawdown" in labels


def test_dashboard_warns_when_no_backtest_export(dashboard_env, tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.config.BACKTEST_RESULTS_PATH", tmp_path / "does_not_exist.json")

    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    assert not at.exception
    assert any("backtest" in w.value.lower() for w in at.warning)


def test_dashboard_errors_clearly_on_stale_export_schema(dashboard_env, tmp_path, monkeypatch):
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"summary": {"strategy_name": "old"}, "trades": []}))

    monkeypatch.setattr("quant_platform.config.BACKTEST_RESULTS_PATH", stale)
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()

    assert not at.exception, [e.message for e in at.exception]
    assert at.error, "a stale export schema must produce a friendly error, not a crash"
    assert "unrecognized" in at.error[0].value.lower()


def test_dashboard_errors_clearly_when_tables_missing(db_path, monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()

    assert not at.exception
    assert at.error, "an empty database must produce a helpful error, not a crash"
    assert "ingest" in at.error[0].value
