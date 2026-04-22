# AGENTS.md

## Commands
- Install deps: `uv sync`
- Run the VX data pipeline: `uv run squid-replication`
- Refresh cached raw data: `uv run squid-replication --refresh`
- Use a non-default data root: `uv run squid-replication --data-dir <path>`
- Equivalent module entrypoint: `uv run python -m squid_replication`
- Run all tests: `uv run pytest`
- Run one test file or test: `uv run pytest tests/test_signals.py` or `uv run pytest tests/test_signals.py -k weekly`
- Coverage: `uv run pytest --cov=squid_replication`

## Repo Shape
- Python package code lives in `src/squid_replication/`.
- `src/squid_replication/cboe_vx.py` is the real CLI/data-pipeline entrypoint behind the `squid-replication` script.
- `src/squid_replication/signals.py` holds the notebook-facing signal, return, weighting, and performance helpers.
- `notebooks/signal_test.ipynb` is the main analysis artifact; it is not a thin demo and depends on locally generated parquet data.
- `workbench.py` is ad hoc exploration code, not a supported entrypoint.

## Data And Notebook Gotchas
- Run the VX pipeline at least once before opening the notebook; it expects local parquet outputs under `data/clean/vix_futures/`.
- Generated datasets in `data/raw/`, `data/clean/`, and `data/derived/` are gitignored and should be treated as rebuildable local artifacts.
- `reference/` is also gitignored working material, so do not rely on it for committed changes.
- The bundled non-generated input that the notebook uses is `data/external/spvxtstr_normalized.csv`.

## Tooling Reality
- This repo currently has no repo-local lint, formatter, typecheck, pre-commit, or CI workflow config checked in. Do not invent `ruff`, `mypy`, or GitHub Actions steps in instructions unless you add that tooling.
- `pyproject.toml` is the source of truth for setup: Python `>=3.12`, package manager/build tool `uv`, and dev dependencies for Jupyter plus pytest.

## Verified Current State
- `uv run pytest -q` passes (`34 passed`).
