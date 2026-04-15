# squid-replication

Monthly VX futures scraper and normalizer built with `uv`.

Clean parquet datasets are built with `pandas` and `pyarrow`.

## What it downloads

- Archive monthly VX contract files from the Cboe settlement archive for contracts through `2013`
- Current monthly VX contract files from the Cboe historical futures API for contracts from `2014+`
- `cfevoloi.csv` for daily VX product-level volume and open interest

Weekly VX contracts are ignored.

## Setup

```bash
uv sync
```

## Run

```bash
uv run squid-replication
```

Optional refresh:

```bash
uv run squid-replication --refresh
```

Optional custom output directory:

```bash
uv run squid-replication --data-dir data
```

You can also run the module directly:

```bash
uv run python -m squid_replication
```

## Output layout

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

## Clean datasets

`monthly_contracts.parquet`

- `trade_date`
- `contract_expiry`
- `contract_code`
- `contract_label`
- `source`
- `source_file`
- `open`
- `high`
- `low`
- `close`
- `settle`
- `change`
- `total_volume`
- `efp`
- `open_interest`

Monthly contract cleaning notes:

- Prices before `2007-03-26` are rescaled by `0.1` to normalize the CFE VX quote-scale change
- Archive and current placeholder rows with all-zero prices, volume, and open interest are excluded
- Rows with non-positive `settle` are excluded so the cleaned dataset remains settle-backtestable

`product_daily.parquet`

- `trade_date`
- `vx_volume`
- `vx_open_interest`

`generic_contracts.parquet`

- `trade_date`
- `ux_symbol`
- `ux_rank`
- `contract_expiry`
- `contract_code`
- `contract_label`
- `source`
- `source_file`
- `roll_trade_date`
- `days_to_expiry`
- `open`
- `high`
- `low`
- `close`
- `settle`
- `change`
- `total_volume`
- `efp`
- `open_interest`
- `previous_contract_code`
- `rolled_today`
- `gross_return`
- `transaction_cost`
- `net_return`
- `gross_index`
- `net_index`

Generic VX notes:

- `UX1` through `UX7` are assigned by nearest eligible monthly expiry on each `trade_date`
- A contract stops being eligible on the roll date, defined as `3` trading sessions before `contract_expiry`
- On a roll date, the series is marked on the prior held contract's `settle` and then switches into the next contract at that same `settle`
- `transaction_cost` is a fixed `0.0002` and is applied only when the series switches contracts
- `gross_index` and `net_index` are settle-to-settle wealth indices built from those roll-aware returns
- Later generics use staggered starts and begin once enough eligible months exist

## Tests

```bash
uv run pytest --cov=squid_replication
```
