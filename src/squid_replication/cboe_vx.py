"""Build a local VX futures dataset from Cboe historical files.

This module downloads archived and current monthly VX contract CSVs together
with the daily CFE product volume/open-interest file, normalizes the raw data,
and writes monthly, product-level, and generic continuous parquet datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import pandas as pd

ARCHIVE_CHUNK_PATTERN = re.compile(
    r'(?P<path>/[^"\']*settlement-archive/page-[^"\']+\.js)'
)
ARCHIVE_FILE_PATTERN = re.compile(r"CFE_([FGHJKMNQUVXZ])(\d{2})_VX\.csv")

MONTH_CODE_TO_NUMBER = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}
NUMBER_TO_MONTH_CODE = {value: key for key, value in MONTH_CODE_TO_NUMBER.items()}

CONTRACT_COLUMNS = [
    "trade_date",
    "contract_expiry",
    "contract_code",
    "contract_label",
    "source",
    "source_file",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "change",
    "total_volume",
    "efp",
    "open_interest",
]

PRODUCT_COLUMNS = [
    "trade_date",
    "vx_volume",
    "vx_open_interest",
]

GENERIC_COLUMNS = [
    "trade_date",
    "ux_symbol",
    "ux_rank",
    "contract_expiry",
    "contract_code",
    "contract_label",
    "source",
    "source_file",
    "roll_trade_date",
    "days_to_expiry",
    "open",
    "high",
    "low",
    "close",
    "close_expiry_roll",
    "settle",
    "settle_expiry_roll",
    "change",
    "total_volume",
    "efp",
    "open_interest",
    "previous_contract_code",
    "rolled_today",
    "gross_return",
    "transaction_cost",
    "net_return",
    "gross_index",
    "net_index",
]

MAX_GENERIC_MONTHS = 7
ROLL_TRADING_DAYS = 3
TRANSACTION_COST_BPS = 2.0
VX_SCALE_CHANGE_DATE = date(2007, 3, 26)
VX_PRE_SCALE_FACTOR = 0.1
PRICE_FIELDS = ("open", "high", "low", "close", "settle", "change")
PLACEHOLDER_PRICE_FIELDS = ("open", "high", "low", "close", "settle")


@dataclass(frozen=True)
class SourceConfig:
    archive_page_url: str = "https://www.cboe.com/markets/us/futures/market-statistics/historical-data/settlement-archive"
    current_index_url: str = "https://www-api.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/"
    product_daily_url: str = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/cfevoloi.csv"
    archive_cdn_base: str = (
        "https://cdn.cboe.com/resources/futures/archive/volume-and-price/"
    )
    current_cdn_base: str = "https://cdn.cboe.com/"


@dataclass(frozen=True)
class ContractFile:
    source: str
    filename: str
    url: str
    contract_expiry: date

    def to_dict(self) -> dict[str, str]:
        """Serialize a contract file descriptor.

        Convert the contract metadata into a JSON-safe mapping for cache files.

        Args:
            None.

        Returns:
            dict[str, str]: Serialized contract metadata with an ISO expiry date.

        Raises:
            None explicitly.
        """
        return {
            "source": self.source,
            "filename": self.filename,
            "url": self.url,
            "contract_expiry": self.contract_expiry.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "ContractFile":
        """Deserialize a contract file descriptor.

        Build a `ContractFile` instance from cached metadata loaded from JSON.

        Args:
            payload: Serialized contract metadata including source, filename,
                url, and `contract_expiry`.

        Returns:
            ContractFile: Parsed contract file metadata.

        Raises:
            KeyError: If a required metadata field is missing.
            ValueError: If `contract_expiry` uses an unsupported date format.
        """
        return cls(
            source=payload["source"],
            filename=payload["filename"],
            url=payload["url"],
            contract_expiry=parse_date(payload["contract_expiry"]),
        )


@dataclass(frozen=True)
class PipelineResult:
    data_dir: Path
    contract_files: int
    contract_rows: int
    product_rows: int
    generic_rows: int
    raw_dir: Path
    clean_dir: Path


DEFAULT_CONFIG = SourceConfig()


def main() -> None:
    """Run the VX pipeline CLI.

    Parse command line options, execute the data pipeline, and print output
    locations and row counts for the generated datasets.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If argument parsing exits due to invalid input or `--help`.
    """
    parser = argparse.ArgumentParser(
        description="Download and normalize monthly VX futures history from Cboe."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Base directory for raw and clean outputs."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download metadata and raw files even if present.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    inline_status_active = False
    inline_status_width = 0

    def finish_inline_status() -> None:
        nonlocal inline_status_active, inline_status_width
        if not inline_status_active:
            return
        print()
        inline_status_active = False
        inline_status_width = 0

    def status(message: str) -> None:
        finish_inline_status()
        print(f"[vx] {message}", flush=True)

    def progress(current: int, total: int) -> None:
        nonlocal inline_status_active, inline_status_width
        message = f"[vx] Downloading files: {current}/{total}"
        padding = max(0, inline_status_width - len(message))
        print(f"\r{message}{' ' * padding}", end="", flush=True)
        inline_status_active = True
        inline_status_width = len(message)

    status(f"Starting VX futures pipeline in {data_dir}")
    if args.refresh:
        status("Refresh enabled; cached metadata and raw files will be replaced")

    try:
        result = run_pipeline(
            data_dir=data_dir,
            refresh=args.refresh,
            status=status,
            progress=progress,
        )
    finally:
        finish_inline_status()

    print(f"Saved {result.contract_files} monthly contract files to {result.raw_dir}")
    print(
        f"Wrote {result.contract_rows} contract rows to {result.clean_dir / 'monthly_contracts.parquet'}"
    )
    print(
        f"Wrote {result.product_rows} VX daily rows to {result.clean_dir / 'product_daily.parquet'}"
    )
    print(
        f"Wrote {result.generic_rows} generic VX rows to {result.clean_dir / 'generic_contracts.parquet'}"
    )


def run_pipeline(
    data_dir: Path,
    refresh: bool = False,
    client: httpx.Client | None = None,
    config: SourceConfig = DEFAULT_CONFIG,
    status: Callable[[str], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PipelineResult:
    """Build raw and normalized VX futures datasets.

    Download archived and current monthly VX contract files plus daily product
    statistics, normalize them into DataFrames, and write parquet outputs.

    Args:
        data_dir: Base directory for raw downloads and clean parquet outputs.
        refresh: Whether cached metadata and raw files should be re-fetched.
        client: Optional HTTP client to reuse for network requests.
        config: Source URLs and CDN roots for the pipeline.
        status: Optional callback for progress messages during execution.
        progress: Optional callback for file download progress updates.

    Returns:
        PipelineResult: Counts and output paths for the generated datasets.

    Raises:
        httpx.HTTPError: If a remote resource cannot be fetched successfully.
        OSError: If local output files cannot be created or written.
        ValueError: If fetched source data is malformed or cannot be parsed.
    """
    if status is not None:
        status("Preparing raw and clean output directories")

    raw_dir = data_dir / "raw" / "vix_futures"
    clean_dir = data_dir / "clean" / "vix_futures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "archive").mkdir(parents=True, exist_ok=True)
    (raw_dir / "current").mkdir(parents=True, exist_ok=True)

    owns_client = client is None
    http_client = client or httpx.Client(follow_redirects=True, timeout=30.0)

    try:
        if status is not None:
            status("Loading archived monthly contract index")
        archive_contracts = load_archive_contracts(
            http_client, raw_dir / "archive_index.json", refresh, config
        )
        if status is not None:
            status(f"Found {len(archive_contracts)} archived monthly contracts")

        if status is not None:
            status("Loading current monthly contract index")
        current_payload = load_or_fetch_json(
            http_client,
            config.current_index_url,
            raw_dir / "current_index.json",
            refresh,
        )
        current_contracts = filter_current_monthly_contracts(
            current_payload, config.current_cdn_base
        )
        if status is not None:
            status(f"Found {len(current_contracts)} current monthly contracts")

        selected_contracts = sorted(
            [*archive_contracts, *current_contracts],
            key=lambda item: item.contract_expiry,
        )
        contract_rows: list[dict[str, Any]] = []
        total_contracts = len(selected_contracts)

        if status is not None:
            status("Downloading files")

        for index, contract in enumerate(selected_contracts, start=1):
            if progress is not None:
                progress(index, total_contracts)
            raw_path = raw_dir / contract.source / contract.filename
            csv_text = load_or_fetch_text(http_client, contract.url, raw_path, refresh)
            contract_rows.extend(
                parse_contract_csv(
                    csv_text,
                    source=contract.source,
                    source_file=contract.filename,
                    contract_expiry=contract.contract_expiry,
                )
            )

        if status is not None:
            status("Loading daily VX product volume and open interest")
        product_daily_text = load_or_fetch_text(
            http_client, config.product_daily_url, raw_dir / "cfevoloi.csv", refresh
        )
        product_rows = parse_product_daily_csv(product_daily_text)

        if status is not None:
            status("Building normalized contract, product, and generic datasets")
        contract_frame = build_contract_frame(contract_rows)
        product_frame = build_product_frame(product_rows)
        generic_frame = build_generic_contract_frame(contract_frame)

        if status is not None:
            status("Writing parquet outputs")
        write_frame(contract_frame, clean_dir / "monthly_contracts.parquet")
        write_frame(product_frame, clean_dir / "product_daily.parquet")
        write_frame(generic_frame, clean_dir / "generic_contracts.parquet")

        if status is not None:
            status("Pipeline complete")

        return PipelineResult(
            data_dir=data_dir,
            contract_files=len(selected_contracts),
            contract_rows=len(contract_frame),
            product_rows=len(product_frame),
            generic_rows=len(generic_frame),
            raw_dir=raw_dir,
            clean_dir=clean_dir,
        )
    finally:
        if owns_client:
            http_client.close()


def load_archive_contracts(
    client: httpx.Client,
    cache_path: Path,
    refresh: bool,
    config: SourceConfig,
) -> list[ContractFile]:
    """Load or fetch archived monthly contract metadata.

    Read cached archive metadata when available, or scrape the archive page and
    bundle chunk to discover monthly VX contract files and cache the result.

    Args:
        client: HTTP client used for remote requests.
        cache_path: JSON cache file for discovered archive contracts.
        refresh: Whether to bypass the local cache and fetch fresh metadata.
        config: Source URLs and CDN roots for archive discovery.

    Returns:
        list[ContractFile]: Archive contract descriptors ordered by expiry.

    Raises:
        httpx.HTTPError: If archive resources cannot be fetched successfully.
        KeyError: If cached metadata is missing a required field.
        ValueError: If the archive page or cached data cannot be parsed.
    """
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [ContractFile.from_dict(item) for item in payload]

    archive_page = fetch_text(client, config.archive_page_url)
    archive_chunk_path = discover_archive_chunk_path(archive_page)
    archive_chunk_url = urljoin(config.archive_page_url, archive_chunk_path)
    archive_chunk = fetch_text(client, archive_chunk_url)
    contracts = extract_archive_monthly_contracts(
        archive_chunk, config.archive_cdn_base
    )

    write_text(
        cache_path, json.dumps([contract.to_dict() for contract in contracts], indent=2)
    )
    return contracts


def discover_archive_chunk_path(page_html: str) -> str:
    """Extract the settlement archive chunk path from page HTML.

    Search the archive landing page for the bundled JavaScript asset that lists
    archived settlement file names.

    Args:
        page_html: HTML returned by the Cboe settlement archive page.

    Returns:
        str: Relative path to the archive bundle chunk.

    Raises:
        ValueError: If the bundle path cannot be located in the page HTML.
    """
    match = ARCHIVE_CHUNK_PATTERN.search(page_html)
    if match is None:
        raise ValueError(
            "Could not locate the settlement archive chunk URL in the archive page HTML."
        )
    return match.group("path")


def extract_archive_monthly_contracts(
    chunk_text: str,
    archive_cdn_base: str = DEFAULT_CONFIG.archive_cdn_base,
) -> list[ContractFile]:
    """Parse archive contract files from a bundled chunk.

    Scan the JavaScript bundle for monthly VX CSV file names, deduplicate them,
    derive expiry months from the file codes, and build download URLs.

    Args:
        chunk_text: JavaScript bundle text containing archive file references.
        archive_cdn_base: Base URL used to construct archive CSV download URLs.

    Returns:
        list[ContractFile]: Archive contract descriptors ordered by expiry.

    Raises:
        None explicitly.
    """
    seen: set[str] = set()
    contracts: list[ContractFile] = []

    for match in ARCHIVE_FILE_PATTERN.finditer(chunk_text):
        filename = match.group(0)
        if filename in seen:
            continue
        seen.add(filename)
        month_code = match.group(1)
        year = 2000 + int(match.group(2))
        contracts.append(
            ContractFile(
                source="archive",
                filename=filename,
                url=join_url(archive_cdn_base, filename),
                contract_expiry=date(year, MONTH_CODE_TO_NUMBER[month_code], 1),
            )
        )

    return sorted(contracts, key=lambda item: item.contract_expiry)


def filter_current_monthly_contracts(
    payload: dict[str, list[dict[str, Any]]],
    current_cdn_base: str = DEFAULT_CONFIG.current_cdn_base,
) -> list[ContractFile]:
    """Select current monthly VX contracts from the Cboe index.

    Filter the current product index down to monthly VX contracts, skip years
    already covered by the archive source, and convert entries to URLs.

    Args:
        payload: JSON payload keyed by year from the current contract index.
        current_cdn_base: Base URL used to construct current CSV download URLs.

    Returns:
        list[ContractFile]: Current contract descriptors ordered by expiry.

    Raises:
        KeyError: If a required contract field is missing from the payload.
        ValueError: If an expiry date uses an unsupported format.
    """
    contracts: list[ContractFile] = []

    for year_key in sorted(payload):
        for item in payload[year_key]:
            if item.get("futures_root") != "VX":
                continue
            if item.get("duration_type") != "M":
                continue

            contract_expiry = parse_date(item["expire_date"])
            if contract_expiry.year <= 2013:
                continue

            path = item["path"]
            filename = Path(path).name
            contracts.append(
                ContractFile(
                    source="current",
                    filename=filename,
                    url=join_url(current_cdn_base, path),
                    contract_expiry=contract_expiry,
                )
            )

    return sorted(contracts, key=lambda item: item.contract_expiry)


def parse_contract_csv(
    csv_text: str,
    *,
    source: str,
    source_file: str,
    contract_expiry: date,
) -> list[dict[str, Any]]:
    """Parse a monthly VX contract CSV into normalized rows.

    Skip any file preamble, parse the contract table, infer the effective
    expiry date, normalize historical price scaling, and discard placeholder or
    pre-history rows without a positive settlement.

    Args:
        csv_text: Raw CSV text for one monthly VX contract file.
        source: Source label describing where the file was discovered.
        source_file: Original file name for the parsed contract CSV.
        contract_expiry: Expected expiry date derived from file metadata.

    Returns:
        list[dict[str, Any]]: Normalized contract rows for the monthly file.

    Raises:
        ValueError: If the CSV header or required date values cannot be parsed.
    """
    lines = csv_text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Trade Date,")),
        None,
    )
    if header_index is None:
        raise ValueError("Could not find the contract CSV header row.")

    reader = csv.DictReader(StringIO("\n".join(lines[header_index:])))
    rows: list[dict[str, Any]] = []

    for row in reader:
        trade_date_text = (row.get("Trade Date") or "").strip()
        if not trade_date_text:
            continue
        trade_date = parse_date(trade_date_text)
        rows.append(
            {
                "trade_date": trade_date,
                "contract_label": (row.get("Futures") or "").strip(),
                "source": source,
                "source_file": source_file,
                "open": parse_float(row.get("Open")),
                "high": parse_float(row.get("High")),
                "low": parse_float(row.get("Low")),
                "close": parse_float(row.get("Close")),
                "settle": parse_float(row.get("Settle")),
                "change": parse_float(row.get("Change")),
                "total_volume": parse_int(row.get("Total Volume")),
                "efp": parse_int(row.get("EFP")),
                "open_interest": parse_int(row.get("Open Interest")),
            }
        )

    if not rows:
        return rows

    actual_expiry = resolve_contract_expiry(source, contract_expiry, rows)
    contract_code = build_contract_code(actual_expiry)
    for row in rows:
        row["contract_expiry"] = actual_expiry
        row["contract_code"] = contract_code
        normalize_contract_row(row)

    rows = [row for row in rows if not is_placeholder_row(row)]
    first_valid_settle_index = next(
        (index for index, row in enumerate(rows) if is_positive_settle(row["settle"])),
        None,
    )
    if first_valid_settle_index is None:
        return []

    return [
        row
        for row in rows[first_valid_settle_index:]
        if is_positive_settle(row["settle"])
    ]


def parse_product_daily_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parse the daily VX product CSV into normalized rows.

    Locate the daily product table in the downloaded CSV, then extract trade
    date, VX volume, and VX open interest into a stable row structure.

    Args:
        csv_text: Raw CSV text for the daily CFE product statistics file.

    Returns:
        list[dict[str, Any]]: Parsed daily product rows ordered as in the file.

    Raises:
        ValueError: If the CSV header or required date values cannot be parsed.
    """
    lines = csv_text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Date,")), None
    )
    if header_index is None:
        raise ValueError("Could not find the CFE daily product header row.")

    reader = csv.DictReader(StringIO("\n".join(lines[header_index:])))
    rows: list[dict[str, Any]] = []

    for row in reader:
        trade_date_text = (row.get("Date") or "").strip()
        if not trade_date_text:
            continue
        rows.append(
            {
                "trade_date": parse_date(trade_date_text),
                "vx_volume": parse_int(row.get("VOLATILITY INDEX VOLUME")),
                "vx_open_interest": parse_int(row.get("VOLATILITY INDEX OI")),
            }
        )

    return rows


def build_contract_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the normalized monthly contract frame.

    Convert parsed contract rows into a DataFrame with stable columns, normalize
    date values, and sort rows by contract expiry and trade date.

    Args:
        rows: Parsed monthly contract rows.

    Returns:
        pd.DataFrame: Normalized monthly contract dataset.

    Raises:
        None explicitly.
    """
    if not rows:
        return empty_frame(CONTRACT_COLUMNS)
    frame = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
    frame = normalize_date_columns(frame, ["trade_date", "contract_expiry"])
    return frame.sort_values(["contract_expiry", "trade_date"]).reset_index(drop=True)


def build_product_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the normalized product-level frame.

    Convert parsed VX daily product rows into a DataFrame with stable columns,
    normalize date values, and sort rows by trade date.

    Args:
        rows: Parsed daily VX product rows.

    Returns:
        pd.DataFrame: Normalized product-level dataset.

    Raises:
        None explicitly.
    """
    if not rows:
        return empty_frame(PRODUCT_COLUMNS)
    frame = pd.DataFrame(rows, columns=PRODUCT_COLUMNS)
    frame = normalize_date_columns(frame, ["trade_date"])
    return frame.sort_values("trade_date").reset_index(drop=True)


def build_generic_contract_frame(
    contract_frame: pd.DataFrame,
    max_rank: int = MAX_GENERIC_MONTHS,
    roll_trading_days: int = ROLL_TRADING_DAYS,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Build generic VX futures series from contract rows.

    Create UX1 through UXn daily snapshots, compute rolling transitions between
    contracts, and derive gross and net return index levels for each rank.

    Args:
        contract_frame: Normalized monthly contract data.
        max_rank: Maximum generic contract rank to build.
        roll_trading_days: Trading days before expiry when the roll occurs.
        transaction_cost_bps: Cost applied on roll days in basis points.

    Returns:
        pd.DataFrame: Generic VX series with roll and return metadata.

    Raises:
        None explicitly.
    """
    if contract_frame.empty:
        return empty_frame(GENERIC_COLUMNS)

    sorted_rows = contract_frame.sort_values(["source_file", "trade_date"]).to_dict(
        orient="records"
    )
    contracts_by_file: dict[str, list[dict[str, Any]]] = {}
    for row in sorted_rows:
        contracts_by_file.setdefault(row["source_file"], []).append(row)

    all_trade_dates = sorted({row["trade_date"] for row in sorted_rows})
    previous_trade_date = {
        trade_date: all_trade_dates[index - 1] if index > 0 else None
        for index, trade_date in enumerate(all_trade_dates)
    }
    transaction_cost = transaction_cost_bps / 10_000.0
    daily_returns_by_file: dict[str, dict[date, float | None]] = {}
    for source_file, contract_rows in contracts_by_file.items():
        daily_returns = build_contract_daily_returns(contract_rows)
        daily_returns_by_file[source_file] = daily_returns

    generic_rows = build_generic_snapshots(
        contracts_by_file,
        max_rank=max_rank,
        roll_trading_days=roll_trading_days,
        include_roll_trade_date=False,
    )
    expiry_roll_rows = build_generic_snapshots(
        contracts_by_file,
        max_rank=max_rank,
        roll_trading_days=0,
        include_roll_trade_date=True,
    )
    expiry_roll_levels = {
        (row["trade_date"], row["ux_symbol"]): {
            "close_expiry_roll": row["close"],
            "settle_expiry_roll": row["settle"],
        }
        for row in expiry_roll_rows
    }

    if not generic_rows:
        return empty_frame(GENERIC_COLUMNS)

    final_rows: list[dict[str, Any]] = []
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in generic_rows:
        rows_by_symbol.setdefault(row["ux_symbol"], []).append(row)

    for rank in range(1, max_rank + 1):
        ux_symbol = f"UX{rank}"
        series_rows = sorted(
            rows_by_symbol.get(ux_symbol, []), key=lambda row: row["trade_date"]
        )
        previous_row: dict[str, Any] | None = None
        gross_index = 1.0
        net_index = 1.0

        for row in series_rows:
            expected_previous_trade_date = previous_trade_date[row["trade_date"]]
            has_series_gap = (
                previous_row is None
                or expected_previous_trade_date is None
                or previous_row["trade_date"] != expected_previous_trade_date
            )
            expiry_roll_level = expiry_roll_levels.get(
                (row["trade_date"], ux_symbol), {}
            )
            row["close_expiry_roll"] = expiry_roll_level.get("close_expiry_roll")
            row["settle_expiry_roll"] = expiry_roll_level.get("settle_expiry_roll")

            row["previous_contract_code"] = (
                None if has_series_gap else previous_row["contract_code"]
            )

            if has_series_gap:
                row["rolled_today"] = False
                row["gross_return"] = None
                row["transaction_cost"] = 0.0
                row["net_return"] = None
                if previous_row is None:
                    gross_index = 1.0
                    net_index = 1.0
                row["gross_index"] = gross_index
                row["net_index"] = net_index
            else:
                row["rolled_today"] = row["source_file"] != previous_row["source_file"]
                return_source_file = (
                    previous_row["source_file"]
                    if row["rolled_today"]
                    else row["source_file"]
                )
                row["gross_return"] = daily_returns_by_file[return_source_file].get(
                    row["trade_date"]
                )
                row["transaction_cost"] = (
                    transaction_cost if row["rolled_today"] else 0.0
                )
                row["net_return"] = (
                    None
                    if row["gross_return"] is None
                    else row["gross_return"] - row["transaction_cost"]
                )

                if row["gross_return"] is not None:
                    gross_index *= 1.0 + row["gross_return"]
                if row["net_return"] is not None:
                    net_index *= 1.0 + row["net_return"]

                row["gross_index"] = gross_index
                row["net_index"] = net_index

            final_rows.append(row)
            previous_row = row

    frame = pd.DataFrame(final_rows, columns=GENERIC_COLUMNS)
    frame = normalize_date_columns(
        frame, ["trade_date", "contract_expiry", "roll_trade_date"]
    )
    return frame.sort_values(["trade_date", "ux_rank"]).reset_index(drop=True)


def build_generic_snapshots(
    contracts_by_file: dict[str, list[dict[str, Any]]],
    *,
    max_rank: int,
    roll_trading_days: int,
    include_roll_trade_date: bool,
) -> list[dict[str, Any]]:
    """Build daily generic contract snapshots by rank.

    Select contracts that are eligible on each trade date, assign UX ranks by
    relative expiry month, and carry forward roll date metadata per snapshot.

    Args:
        contracts_by_file: Parsed contract rows grouped by source file name.
        max_rank: Maximum generic contract rank to include.
        roll_trading_days: Trading days before expiry when the roll occurs.
        include_roll_trade_date: Whether the roll date itself stays eligible.

    Returns:
        list[dict[str, Any]]: Daily generic contract snapshots.

    Raises:
        None explicitly.
    """
    eligible_by_date: dict[date, list[dict[str, Any]]] = {}

    for contract_rows in contracts_by_file.values():
        contract_trade_dates = [row["trade_date"] for row in contract_rows]
        roll_trade_date = resolve_roll_trade_date(
            contract_trade_dates, roll_trading_days
        )

        for row in contract_rows:
            if include_roll_trade_date:
                if row["trade_date"] > roll_trade_date:
                    continue
            elif row["trade_date"] >= roll_trade_date:
                continue

            eligible_row = dict(row)
            eligible_row["roll_trade_date"] = roll_trade_date
            eligible_row["days_to_expiry"] = (
                row["contract_expiry"] - row["trade_date"]
            ).days
            eligible_by_date.setdefault(row["trade_date"], []).append(eligible_row)

    generic_rows: list[dict[str, Any]] = []
    for trade_date in sorted(eligible_by_date):
        daily_rows = sorted(
            eligible_by_date[trade_date], key=lambda row: row["contract_expiry"]
        )
        if not daily_rows:
            continue

        front_expiry = daily_rows[0]["contract_expiry"]
        for row in daily_rows:
            rank = expiry_month_offset(front_expiry, row["contract_expiry"]) + 1
            if rank > max_rank:
                continue
            generic_rows.append(
                {
                    "trade_date": row["trade_date"],
                    "ux_symbol": f"UX{rank}",
                    "ux_rank": rank,
                    "contract_expiry": row["contract_expiry"],
                    "contract_code": row["contract_code"],
                    "contract_label": row["contract_label"],
                    "source": row["source"],
                    "source_file": row["source_file"],
                    "roll_trade_date": row["roll_trade_date"],
                    "days_to_expiry": row["days_to_expiry"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "settle": row["settle"],
                    "change": row["change"],
                    "total_volume": row["total_volume"],
                    "efp": row["efp"],
                    "open_interest": row["open_interest"],
                }
            )

    return generic_rows


def write_frame(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a DataFrame to parquet.

    Ensure the target directory exists before saving the DataFrame without an
    index column.

    Args:
        frame: DataFrame to serialize.
        output_path: Destination parquet path.

    Returns:
        None.

    Raises:
        ImportError: If no parquet engine is available.
        OSError: If the output path cannot be created or written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def empty_frame(columns: list[str]) -> pd.DataFrame:
    """Create an empty DataFrame with fixed columns.

    Provide a stable empty frame shape for pipeline stages that have no rows.

    Args:
        columns: Column names to assign to the empty frame.

    Returns:
        pd.DataFrame: Empty DataFrame with the requested columns.

    Raises:
        None explicitly.
    """
    return pd.DataFrame(columns=columns)


def normalize_date_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize date-like columns to plain `date` values.

    Copy the frame and coerce selected columns so `datetime` instances become
    `date` objects while null-like values remain unchanged.

    Args:
        frame: DataFrame whose date columns should be normalized.
        columns: Column names to normalize when present.

    Returns:
        pd.DataFrame: Copy of the input frame with normalized date values.

    Raises:
        None explicitly.
    """
    normalized = frame.copy()
    for column in columns:
        if column not in normalized:
            continue
        normalized[column] = normalized[column].map(normalize_date_value)
    return normalized


def normalize_date_value(value: Any) -> Any:
    """Normalize a single date-like value.

    Convert `datetime` instances to calendar dates and leave existing `date`
    objects, null-like values, and unrelated values unchanged.

    Args:
        value: Value to normalize.

    Returns:
        Any: Normalized value.

    Raises:
        None explicitly.
    """
    if pd.isna(value):
        return value
    if isinstance(value, datetime):
        return value.date()
    return value


def load_or_fetch_text(
    client: httpx.Client, url: str, path: Path, refresh: bool
) -> str:
    """Load cached text or fetch it from a URL.

    Return the cached file contents when available unless refresh is requested;
    otherwise fetch the remote text, cache it locally, and return it.

    Args:
        client: HTTP client used for remote requests.
        url: Remote URL for the text resource.
        path: Local cache path for the downloaded text.
        refresh: Whether to bypass an existing cached file.

    Returns:
        str: Text content loaded from cache or fetched remotely.

    Raises:
        httpx.HTTPError: If the remote resource cannot be fetched.
        OSError: If the fetched content cannot be written to disk.
    """
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    text = fetch_text(client, url)
    write_text(path, text)
    return text


def load_or_fetch_json(
    client: httpx.Client, url: str, path: Path, refresh: bool
) -> dict[str, Any]:
    """Load cached JSON or fetch it from a URL.

    Return the cached JSON payload when available unless refresh is requested;
    otherwise fetch the remote payload, cache it locally, and return it.

    Args:
        client: HTTP client used for remote requests.
        url: Remote URL for the JSON resource.
        path: Local cache path for the downloaded payload.
        refresh: Whether to bypass an existing cached file.

    Returns:
        dict[str, Any]: JSON payload loaded from cache or fetched remotely.

    Raises:
        httpx.HTTPError: If the remote resource cannot be fetched.
        json.JSONDecodeError: If cached or fetched content is not valid JSON.
        OSError: If the fetched payload cannot be written to disk.
    """
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    payload = fetch_json(client, url)
    write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return payload


def fetch_text(client: httpx.Client, url: str) -> str:
    """Fetch text content from a URL.

    Issue a GET request and require a successful HTTP status before returning
    the response body as text.

    Args:
        client: HTTP client used for the request.
        url: Remote URL to fetch.

    Returns:
        str: Response body decoded as text.

    Raises:
        httpx.HTTPError: If the request fails or returns an error status.
    """
    response = client.get(url)
    response.raise_for_status()
    return response.text


def fetch_json(client: httpx.Client, url: str) -> dict[str, Any]:
    """Fetch JSON content from a URL.

    Issue a GET request, require a successful HTTP status, and decode the
    response body as JSON.

    Args:
        client: HTTP client used for the request.
        url: Remote URL to fetch.

    Returns:
        dict[str, Any]: Decoded JSON payload.

    Raises:
        httpx.HTTPError: If the request fails or returns an error status.
        json.JSONDecodeError: If the response body is not valid JSON.
    """
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text to disk.

    Ensure the parent directory exists before writing the full text payload.

    Args:
        path: Destination path for the text file.
        content: UTF-8 text content to write.

    Returns:
        None.

    Raises:
        OSError: If the destination cannot be created or written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def join_url(base: str, path: str) -> str:
    """Join a base URL and relative path safely.

    Normalize leading and trailing slashes so URL segments combine without
    duplicate or missing separators.

    Args:
        base: Base URL prefix.
        path: Relative or absolute URL path segment.

    Returns:
        str: Joined URL.

    Raises:
        None explicitly.
    """
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def parse_date(value: str) -> date:
    """Parse a supported date string into a `date`.

    Accept both ISO dates and the slash-delimited dates used by Cboe CSV files.

    Args:
        value: Date string to parse.

    Returns:
        date: Parsed calendar date.

    Raises:
        ValueError: If the input does not match a supported date format.
    """
    for pattern in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value!r}")


def parse_float(value: str | None) -> float | None:
    """Parse an optional numeric string as `float`.

    Strip surrounding whitespace and treat missing or empty strings as nulls.

    Args:
        value: Optional text representation of a numeric value.

    Returns:
        float | None: Parsed floating-point value or `None` for empty input.

    Raises:
        ValueError: If the text is non-empty but not a valid float.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def parse_int(value: str | None) -> int | None:
    """Parse an optional numeric string as `int`.

    Strip surrounding whitespace and treat missing or empty strings as nulls.
    Non-empty inputs are parsed through `float` first to handle CSV numerics.

    Args:
        value: Optional text representation of a numeric value.

    Returns:
        int | None: Parsed integer value or `None` for empty input.

    Raises:
        ValueError: If the text is non-empty but not a valid number.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return int(float(text))


def build_contract_code(contract_expiry: date) -> str:
    """Build a CFE month code for a contract expiry.

    Translate the expiry month into the standard futures month letter and append
    the final two digits of the expiry year.

    Args:
        contract_expiry: Expiry date used to derive the month code.

    Returns:
        str: Two-part contract code such as `F24`.

    Raises:
        None explicitly.
    """
    return (
        f"{NUMBER_TO_MONTH_CODE[contract_expiry.month]}{str(contract_expiry.year)[-2:]}"
    )


def resolve_contract_expiry(
    source: str, contract_expiry: date, rows: list[dict[str, Any]]
) -> date:
    """Resolve the effective expiry date for a contract file.

    Keep the provided expiry date for current-source files, but use the final
    trade date for archive files whose metadata only identifies the month.

    Args:
        source: Contract source label such as `archive` or `current`.
        contract_expiry: Expiry date supplied by the source metadata.
        rows: Parsed rows for the contract file.

    Returns:
        date: Effective contract expiry date.

    Raises:
        ValueError: If `rows` is empty for an archive contract.
    """
    if source == "current":
        return contract_expiry
    return max(row["trade_date"] for row in rows)


def resolve_roll_trade_date(trade_dates: list[date], roll_trading_days: int) -> date:
    """Pick the trade date when a generic series should roll.

    Use the date that is `roll_trading_days` sessions before the final trade
    date, or the first available date when the history is shorter.

    Args:
        trade_dates: Ordered trade dates for one contract history.
        roll_trading_days: Trading days before expiry when the roll occurs.

    Returns:
        date: Trade date on which the generic series should roll.

    Raises:
        IndexError: If `trade_dates` is empty.
    """
    if len(trade_dates) <= roll_trading_days:
        return trade_dates[0]
    return trade_dates[-(roll_trading_days + 1)]


def expiry_month_offset(front_expiry: date, contract_expiry: date) -> int:
    """Compute the month distance from front to target expiry.

    Express the offset in whole calendar months so nearby contracts can be
    mapped to UX ranks relative to the front contract.

    Args:
        front_expiry: Expiry date of the front contract.
        contract_expiry: Expiry date of the contract being ranked.

    Returns:
        int: Month offset from `front_expiry` to `contract_expiry`.

    Raises:
        None explicitly.
    """
    return (
        (contract_expiry.year - front_expiry.year) * 12
        + contract_expiry.month
        - front_expiry.month
    )


def build_contract_daily_returns(
    contract_rows: list[dict[str, Any]],
) -> dict[date, float | None]:
    """Calculate daily settle-to-settle returns for a contract.

    Walk the contract history in order and compute percentage changes in the
    settlement price, returning nulls when either side of the comparison is
    missing or zero.

    Args:
        contract_rows: Ordered rows for a single contract history.

    Returns:
        dict[date, float | None]: Daily return keyed by trade date.

    Raises:
        None explicitly.
    """
    daily_returns: dict[date, float | None] = {}
    previous_settle: float | None = None

    for row in contract_rows:
        settle = row["settle"]
        if previous_settle in (None, 0.0) or settle in (None, 0.0):
            daily_returns[row["trade_date"]] = None
        else:
            daily_returns[row["trade_date"]] = settle / previous_settle - 1.0
        previous_settle = settle

    return daily_returns


def normalize_contract_row(row: dict[str, Any]) -> None:
    """Apply historical price scaling to a contract row.

    Rescale pre-change VX price fields so older rows are comparable with the
    post-change contract convention used in newer source files.

    Args:
        row: Parsed contract row to normalize in place.

    Returns:
        None.

    Raises:
        KeyError: If a required price field is missing from `row`.
    """
    if row["trade_date"] < VX_SCALE_CHANGE_DATE:
        for field in PRICE_FIELDS:
            if row[field] is not None:
                row[field] *= VX_PRE_SCALE_FACTOR


def is_placeholder_row(row: dict[str, Any]) -> bool:
    """Check whether a parsed row is an all-zero placeholder.

    Treat rows with no prices, no volume, and no open interest as placeholders
    that should be removed from the normalized contract history.

    Args:
        row: Parsed contract row to inspect.

    Returns:
        bool: `True` when the row is a placeholder entry.

    Raises:
        KeyError: If a required field is missing from `row`.
    """
    all_prices_zero = all(
        (row[field] or 0.0) == 0.0 for field in PLACEHOLDER_PRICE_FIELDS
    )
    no_activity = (row["total_volume"] or 0) == 0 and (row["open_interest"] or 0) == 0
    return all_prices_zero and no_activity


def is_positive_settle(settle: float | None) -> bool:
    """Check whether a settlement value is positive.

    Provide a small predicate for filtering out empty or non-positive settle
    values when building normalized histories.

    Args:
        settle: Settlement value to evaluate.

    Returns:
        bool: `True` when the settlement is a positive number.

    Raises:
        None explicitly.
    """
    return settle is not None and settle > 0.0
