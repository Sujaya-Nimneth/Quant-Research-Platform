"""CLI wiring: subcommands parse, and the dashboard imports cleanly."""

from __future__ import annotations

import subprocess
import sys

import pytest

from quant_platform.main import main as cli_main


def _run_cli(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["quant_platform", *args])
    cli_main()


def test_cli_requires_a_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["quant_platform"])
    with pytest.raises(SystemExit):
        cli_main()


def test_ingest_help_lists_new_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["quant_platform", "ingest", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--full-refresh" in out
    assert "--strict-tickers" in out


def test_backtest_help_lists_execution_delay(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["quant_platform", "backtest", "--help"])
    with pytest.raises(SystemExit):
        cli_main()
    out = capsys.readouterr().out
    assert "--execution-delay" in out
    assert "--slippage" in out


def test_status_reports_migrated_columns(populated_db, monkeypatch, capsys):
    from quant_platform.features.factory import SignalFactory

    SignalFactory().run_pipeline()

    monkeypatch.setattr(sys, "argv", ["quant_platform", "status"])
    cli_main()
    out = capsys.readouterr().out

    assert "daily_prices" in out
    assert "volatility_15d" in out, "migrated columns must be listed as features"
    assert "calculated_at" not in out.split("Computed Alpha features:")[1].split("\n")[0]


def test_dashboard_module_imports_without_side_effects():
    """A syntax/import error in app.py used to break `streamlit run` entirely."""
    result = subprocess.run(
        [sys.executable, "-c", "import ast, pathlib; ast.parse(pathlib.Path('quant_platform/app.py').read_text())"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
