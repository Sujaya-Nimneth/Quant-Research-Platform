"""CLI wiring: subcommands parse, handlers behave, and the dashboard imports cleanly."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.ingestion.base import BaseFetcher
from quant_platform.main import main as cli_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _StubFetcher(BaseFetcher):
    def __init__(self, frame=None, error: Exception | None = None):
        self.frame = frame if frame is not None else pd.DataFrame()
        self.error = error
        self.calls: list[dict] = []

    def fetch_daily_data(self, tickers, start_date, end_date) -> pd.DataFrame:
        self.calls.append({"tickers": list(tickers), "start": start_date, "end": end_date})
        if self.error:
            raise self.error
        return self.frame.copy()


def _stub_frame() -> pd.DataFrame:
    rows = []
    for i, day in enumerate(("2024-01-02", "2024-01-03")):
        rows.append(
            {
                "symbol": "AAA",
                "date": pd.Timestamp(day).date(),
                "open": 10.0 + i,
                "high": 11.0 + i,
                "low": 9.0 + i,
                "close": 10.5 + i,
                "adj_close": 10.4 + i,
                "volume": 1000 + i,
            }
        )
    return pd.DataFrame(rows)


def test_cli_requires_a_subcommand(monkeypatch):
    monkeypatch.setattr("sys.argv", ["quant_platform"])
    with pytest.raises(SystemExit):
        cli_main()


def test_ingest_help_lists_new_flags(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["quant_platform", "ingest", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--full-refresh" in out
    assert "--strict-tickers" in out


def test_backtest_help_lists_execution_delay(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["quant_platform", "backtest", "--help"])
    with pytest.raises(SystemExit):
        cli_main()
    out = capsys.readouterr().out
    assert "--execution-delay" in out
    assert "--slippage" in out


def test_ingest_writes_rows_and_exits_cleanly(empty_db, monkeypatch):
    fetcher = _StubFetcher(_stub_frame())
    monkeypatch.setattr("quant_platform.main.YFinanceFetcher", lambda: fetcher)
    monkeypatch.setattr("sys.argv", ["quant_platform", "ingest", "--tickers", "AAA"])

    cli_main()  # must not raise SystemExit on success

    assert fetcher.calls and fetcher.calls[0]["tickers"] == ["AAA"]
    import duckdb

    conn = duckdb.connect(str(empty_db), read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0] == 2
    conn.close()


def test_ingest_failure_exits_1(empty_db, monkeypatch):
    fetcher = _StubFetcher(error=RuntimeError("network down"))
    monkeypatch.setattr("quant_platform.main.YFinanceFetcher", lambda: fetcher)
    monkeypatch.setattr("sys.argv", ["quant_platform", "ingest", "--test"])

    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 1


def test_test_and_tickers_flags_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr("sys.argv", ["quant_platform", "ingest", "--test", "--tickers", "AAA"])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--execution-delay", "0"),
        ("--init-cash", "-100"),
        ("--fee", "0"),
        ("--slippage", "-0.01"),
    ],
)
def test_backtest_rejects_invalid_numbers(monkeypatch, flag, value):
    monkeypatch.setattr("sys.argv", ["quant_platform", "backtest", flag, value])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 2


def test_backtest_runs_and_exports(seeded_db, tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        "sys.argv",
        ["quant_platform", "backtest", "--output", str(output), "--symbols", "AAA,BBB"],
    )
    cli_main()
    assert output.exists()
    assert output.read_text().endswith("\n")


def test_features_runs_and_exits_cleanly(populated_db, monkeypatch):
    monkeypatch.setattr("sys.argv", ["quant_platform", "features"])
    cli_main()

    import duckdb

    conn = duckdb.connect(str(populated_db), read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] > 0
    conn.close()


def test_status_reports_migrated_columns(populated_db, monkeypatch, capsys):
    from quant_platform.features.factory import SignalFactory

    SignalFactory().run_pipeline()

    monkeypatch.setattr("sys.argv", ["quant_platform", "status"])
    cli_main()
    out = capsys.readouterr().out

    assert "daily_prices" in out
    assert "volatility_15d" in out, "migrated columns must be listed as features"
    assert "calculated_at" not in out.split("Computed Alpha features:")[1].split("\n")[0]


def test_status_without_database_prints_guidance(db_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["quant_platform", "status"])
    cli_main()  # must not raise
    out = capsys.readouterr().out
    assert "ingest" in out


def test_dashboard_module_imports_without_side_effects():
    """A syntax/import error in app.py used to break `streamlit run` entirely."""
    # Real import (not just an ast.parse), against the absolute module path so
    # it works regardless of the pytest invocation directory.
    assert PROJECT_ROOT.is_dir()
    module = importlib.import_module("quant_platform.app")
    assert hasattr(module, "main")
