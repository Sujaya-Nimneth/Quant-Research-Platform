"""End-to-end runner: works from any cwd and reports failures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_runner_help_works_from_another_directory(tmp_path):
    """run_all used to fail with ModuleNotFoundError outside the repo root."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "quant_platform" / "run_all.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "--full-universe" in result.stdout
    assert "--full-refresh" in result.stdout


def test_module_cli_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "-m", "quant_platform.main", "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr
    for command in ("ingest", "features", "status", "backtest"):
        assert command in result.stdout


def test_run_all_orchestrates_three_steps(monkeypatch):
    from quant_platform import run_all as runner

    calls: list[list[str]] = []

    def fake_run_step(label, args, echo_stdout=False, timeout=None):
        calls.append(list(args))

    monkeypatch.setattr(runner, "_run_step", fake_run_step)
    runner.run_all(test_mode=True)

    assert len(calls) == 3
    ingest_args = calls[0]
    assert ingest_args[:2] == ["quant_platform.main", "ingest"]
    assert "--test" in ingest_args
    assert "--strict-tickers" not in ingest_args


def test_run_all_full_universe_uses_strict_tickers_and_start_date(monkeypatch):
    from quant_platform import run_all as runner

    calls: list[list[str]] = []

    def fake_run_step(label, args, echo_stdout=False, timeout=None):
        calls.append(list(args))

    monkeypatch.setattr(runner, "_run_step", fake_run_step)
    runner.run_all(test_mode=False, start_date="2023-06-01")

    ingest_args = calls[0]
    assert "--test" not in ingest_args
    assert "--strict-tickers" in ingest_args
    assert "--start-date" in ingest_args
    assert ingest_args[ingest_args.index("--start-date") + 1] == "2023-06-01"


def test_run_step_surfaces_failure_and_exits(monkeypatch):
    import subprocess as subprocess_module

    from quant_platform import run_all as runner

    def boom(*args, **kwargs):
        raise subprocess_module.CalledProcessError(returncode=3, cmd="python -m x", output="", stderr="boom")

    monkeypatch.setattr(runner.subprocess, "run", boom)
    with pytest.raises(SystemExit) as exc:
        runner._run_step("step", ["quant_platform.main", "features"])
    assert exc.value.code == 1


def test_run_step_timeout_exits(monkeypatch):
    import subprocess as subprocess_module

    from quant_platform import run_all as runner

    def hang(*args, **kwargs):
        raise subprocess_module.TimeoutExpired(cmd="python -m x", timeout=60)

    monkeypatch.setattr(runner.subprocess, "run", hang)
    with pytest.raises(SystemExit) as exc:
        runner._run_step("step", ["quant_platform.main", "features"], timeout=60)
    assert exc.value.code == 1
