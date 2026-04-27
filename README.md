# squid-replication

Work in progress: this repo is actively being built out, and the replication notebook, strategy logic, and fit-to-paper results are still evolving.

## What This Repo Is

This repository is for replicating JungleRocks's whitepaper around the VIX term-structure signal behind the Squid family of strategies.

Today the repo has two main pieces:

- a local data pipeline that downloads and normalizes monthly VX futures history from Cboe
- a notebook, `notebooks/signal_test.ipynb`, that walks through the whitepaper signal tests, proxy comparisons, base strategies, refined strategies, and parameter calibration

The current workflow is:

1. build the local VX datasets with the CLI
2. open the notebook
3. reproduce the paper's signal-section figures and tables using:
   - official Cboe VX futures data
   - official Cboe spot VIX history
   - bundled `SPVXTSTR` history
   - Yahoo `ES=F` and `SPY` as equity proxies

## Current Status

What is already implemented:

- monthly VX futures download and cleaning
- generic UX ladder construction (`UX1` through `UX7`)
- dislocation-count signal generation
- same-day and `t+1` bucket studies
- bundled normalized `SPVXTSTR` history
- base and refined Squid strategy weights
- notebook plots and performance tables for the current replication pass

What is still imperfect:

- the equity leg is still proxy-based (`ES=F` and `SPY`), not a fully reconstructed historical ES roll series
- some strategy details are still being calibrated against the paper
- alternate PnL constructions are still being tested in the notebook

## Data Sources

The current notebook and pipeline use:

- Cboe settlement archive and current historical futures API for monthly VX contracts
- Cboe `cfevoloi.csv` for VX product-level volume and open interest
- Cboe spot VIX history CSV
- bundled `data/external/spvxtstr_normalized.csv`
- Yahoo Finance for `ES=F` and `SPY`

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

## Build The Local VX Datasets

Generate the raw and clean VX datasets:

```bash
uv run squid-replication
```

Optional refresh:

```bash
uv run squid-replication --refresh
```

Optional custom data directory:

```bash
uv run squid-replication --data-dir data
```

You can also run the module directly:

```bash
uv run python -m squid_replication
```

The notebook expects the clean VX parquet outputs to exist locally, so run the pipeline at least once before opening the notebook.

## Open The Notebook

Start Jupyter:

```bash
uv run jupyter lab
```

Then open:

```text
notebooks/signal_test.ipynb
```

The notebook currently covers:

- signal construction from the VX curve
- term-structure snapshot plots
- same-day and `t+1` bucket tests
- `SPVXTSTR` comparison
- base strategy curves and stats
- refined strategy curves and stats
- proxy comparison and threshold calibration
- exported strategy weights

## Generated Local Outputs

Running the VX pipeline creates local datasets like:

```text
data/
  raw/
    vix_futures/
      archive/
      current/
      archive_index.json
      current_index.json
      cfevoloi.csv
  clean/
    vix_futures/
      monthly_contracts.parquet
      product_daily.parquet
      generic_contracts.parquet
```

The notebook also writes derived artifacts such as:

```text
data/derived/
  strategy_weights.csv
```

These generated folders are gitignored because they can be rebuilt locally.

## Clean VX Datasets

`monthly_contracts.parquet`

- one row per trade date per monthly VX contract
- includes normalized OHLC, settle, volume, and open interest fields

Cleaning rules include:

- pre-`2007-03-26` VX prices are rescaled by `0.1`
- placeholder rows with all-zero prices and activity are removed
- rows with non-positive `settle` are removed

`product_daily.parquet`

- daily VX product-level volume and open interest

`generic_contracts.parquet`

- generic `UX1` through `UX7` ladder
- roll metadata
- roll-aware generic returns and indices
- alternate `close_expiry_roll` and `settle_expiry_roll` levels for hold-through-expiry analysis

Generic ladder behavior:

- `UX1` through `UX7` are assigned by monthly expiry offset from the front eligible contract
- the default roll date is `3` trading sessions before expiry
- `close_expiry_roll` and `settle_expiry_roll` keep hold-through-expiry versions of the ladder for sensitivity checks
- `net_return` includes the fixed transaction-cost treatment used by the generic builder

## Notebook Conventions

The notebook currently separates signal generation from PnL construction.

Examples:

- the paper-facing notebook default now uses `close` for the signal, plus stable tie handling in the dislocation count
- the VX sleeve default uses generic `net_return` with the first and third available listed contracts for paper-style `UX1` and `UX3` exposure
- VX PnL can still be switched to alternate level-based experiments
- `SPVXTSTR` is read from the bundled normalized file
- the equity side is still tested through proxies

Those choices are documented in the notebook itself and may continue to change while the replication is refined.

## Tests

Run the full test suite with:

```bash
uv run pytest
```

Coverage run:

```bash
uv run pytest --cov=squid_replication
```
