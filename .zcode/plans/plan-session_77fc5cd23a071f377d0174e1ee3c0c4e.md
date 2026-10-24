# Quant Research Platform — Improvement Plan

Audit found **4 high-severity issues, ~20 medium, ~30 small** across correctness, robustness, performance, tests, CI, and hygiene. The strongest parts (look-ahead-bias defense, Wilder RSI seeding, factory validation, hermetic tests) stay untouched. Numbers changing after fixes is accepted.

## Phase 1 — Correctness & breakage fixes (high severity)

1. **Use adjusted prices downstream** — `data_access.load_aligned_matrices` (`data_access.py:69`) reads `close` although `adj_close` is ingested specifically to be the canonical adjusted series. Switch the SELECT to `adj_close AS close` so signals, backtest fills, and the dashboard benchmark all run on split/dividend-adjusted prices. Existing DBs already store the column, so no re-ingest is needed; users re-run `features` + `backtest`.
2. **Validate SQL identifiers in data_access** (`data_access.py:71-72`) — indicator names are f-string-interpolated into SQL. Validate each against the same identifier whitelist the factory uses (`factory._quote_identifier`) and raise `ValueError` otherwise.
3. **Streamlit floor** — `app.py:54` uses `width="stretch"` (needs ≥1.46) but `pyproject.toml:19` and `requirements.txt` declare `>=1.40`. Raise both floors to `streamlit>=1.46,<2`.
4. **Single-source the version** — `pyproject.toml` says 0.2.0, `quant_platform/__init__.py` says 0.1.0. Use `dynamic = ["version"]` + `[tool.setuptools.dynamic]` reading `__version__`, and bump `__version__` to 0.2.0.
5. **Volatility column naming** (`volatility.py:43-44`) — columns are hard-coded `volatility_15d` regardless of the `window` arg. Derive names via `f"volatility_{self._window}d"` (default 15 keeps current schema, no migration).

## Phase 2 — Robustness & correctness hardening

**Metrics** (`metrics.py`): Sharpe/Sortino return `inf` (not 0.0) when std is 0 and mean excess > 0; strip `±inf` from pct-change returns; CAGR computes years from the DatetimeIndex elapsed time (fallback to bar count); `equal_weight_benchmark` normalizes each symbol from its first valid value and warns about dropped symbols.

**Data access**: delete the dead alignment loop (lines 86-88 — rebound `matrix` is never stored), replace with one explicit index/column intersection + reindex; single connection per call; pre-check requested indicator columns against `information_schema.columns` with an actionable error naming the strategy.

**Ingestion** (`pipeline.py`, `yfinance_fetcher.py`): per-symbol high-water marks for incremental fetch (a global MAX(date) permanently strands lagging symbols); `end_date` defaults to tomorrow (yfinance `end` is exclusive); guard the rollback in exception paths against raising; count empty-download symbols as failures in the WARNING summary; warn on the `adj_close = close` fallback.

**Features**: mask MACD warmup rows to NaN (docstring already promises this); re-seed RSI after NaN gaps; `factory.register` warns on duplicate signal names; Bollinger switches to `ddof=0` for reference parity.

**Backtest engine** (`engine.py`): drop all-NaN symbols before cash splitting; exclude open trades (`Exit Timestamp` NaT) from the win rate; source `init_cash`/`fee` defaults from config constants (also stops `pipeline.py`'s duplicate `"2024-01-01"`); atomic JSON export (tmp file + rename); fix `output_path: str | None` hint; remove `drawdown_series` re-export from `__all__`; replace the `print()` tear sheet with `logger` output.

**CLI / app / runner**:
- `main.py`: `status` catches missing DB with the same actionable message `require_tables` gives and builds its sample query from discovered feature columns; `logger.exception` instead of message-only error logs; move `dictConfig` inside `main()`; argparse enforces `--execution-delay >= 1`, positive `--init-cash`/`--fee`, and `--test`/`--tickers` mutual exclusion.
- `app.py`: Trades Auditor KPIs computed from the filtered frame (currently contradict the table when a ticker filter is active); stale-schema backtest JSON caught with a friendly `st.error` instead of a raw KeyError crash; RSI marker thresholds read from the export instead of hard-coded 30/70; guard empty equity curves; drop unused `final_value`.
- `run_all.py`: per-step `subprocess` timeouts (~30 min) with `TimeoutExpired` handling; `--start-date` passthrough; `basicConfig` moved into `main()`.
- `config.py`: remove import-time `mkdir` (explicit `ensure_dirs()` called by entry points); delete dead `MIN_VALID_RSI_PERIODS`/`MIN_VALID_BB_PERIODS` and fix the misleading warmup comment.
- `db/connection.py`: reference `config.DB_PATH` at call time (removes the dual-monkeypatch test fragility); catch DuckDB single-writer lock conflicts with an actionable message; stop double-logging normal query errors.
- `db/schema.py`: reduce `signals` DDL to key columns and let the existing migration add feature columns (removes the duplicate feature knowledge).

## Phase 3 — Performance & flexibility

1. **Vectorized Wilder RSI** — replace the O(n) `.iloc` loop (`mean_reversion.py:95-103`) with rolling-mean seed + `ewm(alpha=1/period, adjust=False)`; verify parity against the existing reference-implementation test.
2. **Feature computation on pivoted wide frames** — the factory's per-symbol groupby + per-signal merge loops become one pivot + vectorized rolling ops across all symbols (~500× fewer interpreted iterations).
3. **Push filters into SQL** — `load_aligned_matrices` gains optional `start`/`end`/`symbols` params; `engine.run` and the CLI expose them (also a flexibility win); factory loads only columns the signals declare.
4. **Batched yfinance downloads** — ticker-list download with per-symbol fallback for failures, replacing ~500 sequential round-trips.
5. **Cash pooling decision** — either switch the engine to one vectorbt portfolio with `cash_sharing=True` grouped columns, or record the static-split conservatism in the results payload. Implement if vectorbt's grouping API supports it cleanly; otherwise document (will note in the PR).

## Phase 4 — Tests, CI & hygiene

**Tests**: CLI handler coverage with stubbed fetcher/factory asserting `SystemExit` codes and side effects; `run_all` orchestration tests (`_run_step` monkeypatched + one failing subprocess → exit 1); fix `test_dashboard_module_imports...` (absolute path, real import — it's currently just a cwd-relative syntax check); make dashboard fixtures module-scoped via `tmp_path_factory` to cut suite runtime; new unit tests for: flat-curve Sharpe→inf, adj_close end-to-end use, SQL identifier rejection, MACD warmup NaN, volatility window naming, misaligned matrices, `load_backtest_results` edge cases.

**CI** (`ci.yml`): add `pip install -e .` plus a `quant-platform --help` smoke step (packaging is currently never exercised); run `pytest --cov=quant_platform --cov-report=term-missing`; add Python 3.14 to the matrix; `concurrency` with `cancel-in-progress`; `workflow_dispatch` trigger.

**Hygiene**: add MIT LICENSE (say so if you prefer another) + `license`/`classifiers`/`urls` in pyproject; `py.typed` marker; `.gitignore` gains `logs/`, `.env`, broader `data/` pattern; README notes the adjusted-price switch (re-run `features` + `backtest`), labels tear-sheet numbers as sample output, and documents the `quant-platform` console script; remove dead code (`RESERVED_COLUMNS` entry, `_run_cli` helper, unused `capsys`); export `VolatilitySignal` from `features/__init__.py` and rename its registry name `custom_volatility` → `volatility_signals`.

## Execution

Each phase lands as one commit with `pytest` + `ruff check` + `ruff format --check` green before moving on. Phases 1–2 change reported metrics (accepted); Phase 3 item 2 is the only structural refactor and is gated on the existing behavioral tests passing unchanged (RSI parity, look-ahead-bias, upsert idempotency).