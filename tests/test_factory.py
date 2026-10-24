"""Signal factory: discovery, validation, schema migration, and upsert safety."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from quant_platform.db.identifiers import quote_identifier
from quant_platform.features.base import BaseSignal
from quant_platform.features.factory import SignalFactory


class BoomSignal(BaseSignal):
    """A signal that always fails, used to prove existing data survives."""

    @property
    def name(self) -> str:
        return "boom"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("simulated failure")


class ShortSignal(BaseSignal):
    """Returns the wrong number of rows."""

    @property
    def name(self) -> str:
        return "short"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.head(2)[["symbol", "date"]].assign(short_signal=1.0)


class NoKeySignal(BaseSignal):
    @property
    def name(self) -> str:
        return "nokey"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"value": range(len(data))})


class ExtraSignal(BaseSignal):
    """A well-behaved custom signal, mirroring the README example."""

    @property
    def name(self) -> str:
        return "extra"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.sort_values("date")
        out = pd.DataFrame({"symbol": df["symbol"], "date": df["date"]})
        out["extra_metric"] = df["close"].rolling(3).std()
        return out


def test_discovers_builtin_signals_without_duplicates():
    factory = SignalFactory()
    assert "momentum_signals" in factory.signals
    assert "mean_reversion_signals" in factory.signals
    assert "custom_volatility" in factory.signals


def test_validation_rejects_wrong_row_count(populated_db):
    factory = SignalFactory()
    factory._signals = {"short": ShortSignal()}

    with pytest.raises(ValueError, match="rows"):
        factory._validate_signal_output("short", "AAA", pd.DataFrame({"symbol": [], "date": []}), 5)


def test_validation_rejects_missing_keys():
    factory = SignalFactory()
    with pytest.raises(ValueError, match="key columns"):
        factory._validate_signal_output("nokey", "AAA", pd.DataFrame({"value": [1]}), 1)


def test_failing_signal_does_not_wipe_existing_columns(populated_db):
    factory = SignalFactory()
    assert factory.run_pipeline() > 0

    conn = duckdb.connect(str(populated_db), read_only=True)
    before = conn.execute(
        "SELECT symbol, date, rsi, volatility_15d FROM signals WHERE rsi IS NOT NULL ORDER BY symbol, date LIMIT 5"
    ).fetchall()
    conn.close()
    assert before, "the baseline pipeline must produce populated rows"

    factory.register(BoomSignal())
    factory.run_pipeline()

    conn = duckdb.connect(str(populated_db), read_only=True)
    after = conn.execute(
        "SELECT symbol, date, rsi, volatility_15d FROM signals WHERE rsi IS NOT NULL ORDER BY symbol, date LIMIT 5"
    ).fetchall()
    conn.close()

    assert after == before, "a failed signal must not null out existing values"


def test_new_signal_triggers_schema_migration(populated_db):
    factory = SignalFactory()
    factory._signals = {"extra": ExtraSignal()}
    factory.run_pipeline()

    conn = duckdb.connect(str(populated_db), read_only=True)
    columns = [row[1] for row in conn.execute("PRAGMA table_info('signals')").fetchall()]
    values = conn.execute("SELECT COUNT(extra_metric) FROM signals").fetchone()[0]
    conn.close()

    assert "extra_metric" in columns
    assert values > 0


def test_rerun_is_idempotent(populated_db):
    factory = SignalFactory()
    first = factory.run_pipeline()
    second = factory.run_pipeline()

    conn = duckdb.connect(str(populated_db), read_only=True)
    total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    duplicates = conn.execute(
        "SELECT COUNT(*) FROM (SELECT symbol, date FROM signals GROUP BY symbol, date HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    conn.close()

    assert first == second
    assert duplicates == 0
    assert total == first


def test_calculated_at_is_refreshed(populated_db):
    factory = SignalFactory()
    factory.run_pipeline()

    conn = duckdb.connect(str(populated_db), read_only=True)
    first = conn.execute("SELECT MIN(calculated_at) FROM signals").fetchone()[0]
    conn.close()

    factory.run_pipeline()

    conn = duckdb.connect(str(populated_db), read_only=True)
    second = conn.execute("SELECT MIN(calculated_at) FROM signals").fetchone()[0]
    conn.close()

    assert second >= first


def test_unsafe_column_names_are_rejected():
    with pytest.raises(ValueError):
        quote_identifier('evil"; DROP TABLE signals; --')


def test_reserved_columns_are_not_treated_as_features():
    df = pd.DataFrame(columns=["symbol", "date", "calculated_at", "rsi"])
    assert SignalFactory._feature_columns(df) == ["rsi"]
