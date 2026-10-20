"""End-to-end runner: works from any cwd and reports failures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
