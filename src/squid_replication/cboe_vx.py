from __future__ import annotations

import argparse
import csv
import json
import re
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
    "settle",
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
        return {
            "source": self.source,
            "filename": self.filename,
            "url": self.url,
            "contract_expiry": self.contract_expiry.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "ContractFile":
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

    result = run_pipeline(data_dir=Path(args.data_dir), refresh=args.refresh)

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
) -> PipelineResult:
    raw_dir = data_dir / "raw" / "vix_futures"
    clean_dir = data_dir / "clean" / "vix_futures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "archive").mkdir(parents=True, exist_ok=True)
    (raw_dir / "current").mkdir(parents=True, exist_ok=True)

    owns_client = client is None
    http_client = client or httpx.Client(follow_redirects=True, timeout=30.0)

    try:
        archive_contracts = load_archive_contracts(
            http_client, raw_dir / "archive_index.json", refresh, config
        )
        current_payload = load_or_fetch_json(
            http_client,
            config.current_index_url,
            raw_dir / "current_index.json",
            refresh,
        )
        current_contracts = filter_current_monthly_contracts(
            current_payload, config.current_cdn_base
        )

        selected_contracts = sorted(
            [*archive_contracts, *current_contracts],
            key=lambda item: item.contract_expiry,
        )
        contract_rows: list[dict[str, Any]] = []

        for contract in selected_contracts:
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

        product_daily_text = load_or_fetch_text(
            http_client, config.product_daily_url, raw_dir / "cfevoloi.csv", refresh
        )
        product_rows = parse_product_daily_csv(product_daily_text)

        contract_frame = build_contract_frame(contract_rows)
        product_frame = build_product_frame(product_rows)
        generic_frame = build_generic_contract_frame(contract_frame)

        write_frame(contract_frame, clean_dir / "monthly_contracts.parquet")
        write_frame(product_frame, clean_dir / "product_daily.parquet")
        write_frame(generic_frame, clean_dir / "generic_contracts.parquet")

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
    if not rows:
        return empty_frame(CONTRACT_COLUMNS)
    frame = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
    frame = normalize_date_columns(frame, ["trade_date", "contract_expiry"])
    return frame.sort_values(["contract_expiry", "trade_date"]).reset_index(drop=True)


def build_product_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
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

    eligible_by_date: dict[date, list[dict[str, Any]]] = {}
    for source_file, contract_rows in contracts_by_file.items():
        contract_trade_dates = [row["trade_date"] for row in contract_rows]
        roll_trade_date = resolve_roll_trade_date(
            contract_trade_dates, roll_trading_days
        )
        daily_returns = build_contract_daily_returns(contract_rows)
        daily_returns_by_file[source_file] = daily_returns

        for row in contract_rows:
            if row["trade_date"] >= roll_trade_date:
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
        for rank, row in enumerate(daily_rows[:max_rank], start=1):
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


def write_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def normalize_date_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column not in normalized:
            continue
        normalized[column] = normalized[column].map(normalize_date_value)
    return normalized


def normalize_date_value(value: Any) -> Any:
    if pd.isna(value):
        return value
    if isinstance(value, datetime):
        return value.date()
    return value


def load_or_fetch_text(
    client: httpx.Client, url: str, path: Path, refresh: bool
) -> str:
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    text = fetch_text(client, url)
    write_text(path, text)
    return text


def load_or_fetch_json(
    client: httpx.Client, url: str, path: Path, refresh: bool
) -> dict[str, Any]:
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    payload = fetch_json(client, url)
    write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return payload


def fetch_text(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    return response.text


def fetch_json(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def join_url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def parse_date(value: str) -> date:
    for pattern in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value!r}")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return int(float(text))


def build_contract_code(contract_expiry: date) -> str:
    return (
        f"{NUMBER_TO_MONTH_CODE[contract_expiry.month]}{str(contract_expiry.year)[-2:]}"
    )


def resolve_contract_expiry(
    source: str, contract_expiry: date, rows: list[dict[str, Any]]
) -> date:
    if source == "current":
        return contract_expiry
    return max(row["trade_date"] for row in rows)


def resolve_roll_trade_date(trade_dates: list[date], roll_trading_days: int) -> date:
    if len(trade_dates) <= roll_trading_days:
        return trade_dates[0]
    return trade_dates[-(roll_trading_days + 1)]


def build_contract_daily_returns(
    contract_rows: list[dict[str, Any]],
) -> dict[date, float | None]:
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
    if row["trade_date"] < VX_SCALE_CHANGE_DATE:
        for field in PRICE_FIELDS:
            if row[field] is not None:
                row[field] *= VX_PRE_SCALE_FACTOR


def is_placeholder_row(row: dict[str, Any]) -> bool:
    all_prices_zero = all(
        (row[field] or 0.0) == 0.0 for field in PLACEHOLDER_PRICE_FIELDS
    )
    no_activity = (row["total_volume"] or 0) == 0 and (row["open_interest"] or 0) == 0
    return all_prices_zero and no_activity


def is_positive_settle(settle: float | None) -> bool:
    return settle is not None and settle > 0.0
