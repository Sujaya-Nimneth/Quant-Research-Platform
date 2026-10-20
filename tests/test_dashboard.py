"""Dashboard smoke tests using Streamlit's AppTest harness."""

from __future__ import annotations

import json
from pathlib import Path

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


@pytest.fixture()
def dashboard_env(populated_db, tmp_path, monkeypatch):
    """Populate the DB and write a backtest export next to it."""
    from quant_platform.backtest.engine import BacktestEngine
    from quant_platform.backtest.strategy import RSIStrategy
    from quant_platform.features.factory import SignalFactory

    SignalFactory().run_pipeline()

    results_path = tmp_path / "backtest_results.json"
    results = BacktestEngine(RSIStrategy()).run(init_cash=10_000.0)
    results_path.write_text(json.dumps(results))

    # app.py re-imports these names on every rerun, so patch the source module.
    monkeypatch.setattr("quant_platform.config.BACKTEST_RESULTS_PATH", results_path)
    return populated_db


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


def test_dashboard_warns_when_no_backtest_export(populated_db, tmp_path, monkeypatch):
    from quant_platform.features.factory import SignalFactory

    SignalFactory().run_pipeline()
    monkeypatch.setattr("quant_platform.config.BACKTEST_RESULTS_PATH", tmp_path / "does_not_exist.json")

    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    assert not at.exception
    assert any("backtest" in w.value.lower() for w in at.warning)


def test_dashboard_errors_clearly_when_tables_missing(db_path, monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()

    assert not at.exception
    assert at.error, "an empty database must produce a helpful error, not a crash"
    assert "ingest" in at.error[0].value
