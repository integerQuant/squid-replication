from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from squid_replication.cboe_vx import (
    SourceConfig,
    build_generic_contract_frame,
    extract_archive_monthly_contracts,
    filter_current_monthly_contracts,
    parse_contract_csv,
    parse_product_daily_csv,
    run_pipeline,
)


def parquet_date_list(series: pd.Series) -> list[date]:
    return [pd.Timestamp(value).date() for value in series.tolist()]


def test_extract_archive_monthly_contracts_parses_and_sorts_unique_files() -> None:
    chunk = """
    let V={"VX - Cboe S&P 500 Volatility Index (VIX) Futures Price and Volume Detail":{
      2013:[{path:"CFE_G13_VX.csv"},{path:"CFE_F13_VX.csv"}],
      2004:[{path:"CFE_K04_VX.csv"}],
      2013:[{path:"CFE_G13_VX.csv"}]
    }}
    """

    contracts = extract_archive_monthly_contracts(chunk)

    assert [contract.filename for contract in contracts] == [
        "CFE_K04_VX.csv",
        "CFE_F13_VX.csv",
        "CFE_G13_VX.csv",
    ]
    assert contracts[0].contract_expiry == date(2004, 5, 1)
    assert contracts[-1].url.endswith(
        "/resources/futures/archive/volume-and-price/CFE_G13_VX.csv"
    )


def test_filter_current_monthly_contracts_ignores_weekly_and_2013() -> None:
    payload = {
        "2013": [
            {
                "product_display": "VX+VXT/F3",
                "expire_date": "2013-01-16",
                "futures_root": "VX",
                "duration_type": "M",
                "path": "data/us/futures/market_statistics/historical_data/VX/VX_2013-01-16.csv",
            }
        ],
        "2014": [
            {
                "product_display": "VX+VXT/F4",
                "expire_date": "2014-01-22",
                "futures_root": "VX",
                "duration_type": "M",
                "path": "data/us/futures/market_statistics/historical_data/VX/VX_2014-01-22.csv",
            }
        ],
        "2015": [
            {
                "product_display": "VX+VXT32/Q5",
                "expire_date": "2015-08-05",
                "futures_root": "VX",
                "duration_type": "W",
                "path": "data/us/futures/market_statistics/historical_data/VX/VX_2015-08-05.csv",
            }
        ],
    }

    contracts = filter_current_monthly_contracts(payload)

    assert [contract.filename for contract in contracts] == ["VX_2014-01-22.csv"]
    assert contracts[0].contract_expiry == date(2014, 1, 22)
    assert contracts[0].source == "current"


def test_parse_contract_csv_uses_true_archive_expiry_from_final_trade_date() -> None:
    csv_text = """Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest
01/02/2013,F (Jan 13),16.80,16.80,15.50,15.60,15.60,-2.10,97535,3632,120663
01/16/2013,F (Jan 13),0.00,14.25,14.40,0.00,13.69,-0.51,0,0,50051
"""

    rows = parse_contract_csv(
        csv_text,
        source="archive",
        source_file="CFE_F13_VX.csv",
        contract_expiry=date(2013, 1, 1),
    )

    assert rows[0]["trade_date"] == date(2013, 1, 2)
    assert rows[0]["contract_expiry"] == date(2013, 1, 16)
    assert rows[0]["contract_code"] == "F13"
    assert rows[0]["contract_label"] == "F (Jan 13)"
    assert rows[0]["open"] == 16.8
    assert rows[0]["total_volume"] == 97535
    assert rows[1]["settle"] == 13.69
    assert rows[1]["source"] == "archive"


def test_parse_contract_csv_preserves_true_current_expiry_from_metadata() -> None:
    csv_text = """Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest
2026-04-09,K (May 2026),20.10,20.40,19.90,20.00,20.05,0.20,100,0,500
2026-04-10,K (May 2026),20.30,20.60,20.10,20.50,20.55,0.50,120,0,520
"""

    rows = parse_contract_csv(
        csv_text,
        source="current",
        source_file="VX_2026-05-20.csv",
        contract_expiry=date(2026, 5, 20),
    )

    assert [row["contract_expiry"] for row in rows] == [date(2026, 5, 20)] * 2
    assert [row["contract_code"] for row in rows] == ["K26", "K26"]


def test_parse_contract_csv_scales_pre_20070326_prices() -> None:
    csv_text = """Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest
03/23/2007,J (Apr 07),134.00,135.50,133.00,133.50,133.30,-0.30,369,0,10632
03/26/2007,J (Apr 07),13.49,14.50,13.21,13.40,13.38,0.50,675,0,11006
"""

    rows = parse_contract_csv(
        csv_text,
        source="archive",
        source_file="CFE_J07_VX.csv",
        contract_expiry=date(2007, 4, 1),
    )

    assert rows[0]["open"] == 13.4
    assert round(rows[0]["close"], 2) == 13.35
    assert round(rows[0]["settle"], 2) == 13.33
    assert round(rows[0]["change"], 2) == -0.03
    assert rows[1]["open"] == 13.49
    assert rows[1]["settle"] == 13.38


def test_parse_contract_csv_drops_placeholder_rows_and_leading_invalid_settles() -> (
    None
):
    csv_text = """Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest
2013-04-19,F (Jan 2014),0,0,0,0,0,0,0,0,0
2013-04-22,F (Jan 2014),19.6500,19.7000,19.5500,19.5500,0,0,4,0,4
2013-04-23,F (Jan 2014),19.7000,20.2000,19.2000,19.3500,0,0,363,0,303
2013-05-20,F (Jan 2014),18.9000,19.1000,18.8500,18.9800,19.0200,0.0200,48,0,1470
2013-05-21,F (Jan 2014),19.0100,19.1500,18.9700,19.1200,19.1400,0.1200,35,0,1482
"""

    rows = parse_contract_csv(
        csv_text,
        source="current",
        source_file="VX_2014-01-22.csv",
        contract_expiry=date(2014, 1, 22),
    )

    assert [row["trade_date"] for row in rows] == [date(2013, 5, 20), date(2013, 5, 21)]
    assert [row["settle"] for row in rows] == [19.02, 19.14]


def test_parse_contract_csv_skips_archive_disclaimer_and_blank_lines() -> None:
    csv_text = """CFE data is compiled for the convenience of site visitors.

Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest

10/10/2012,N (Jul 13),0.00,0.00,0.00,0.00,0.00,0.00,0,0,0

10/22/2012,N (Jul 13),25.00,25.00,24.45,24.45,24.40,24.40,30,0,30

10/23/2012,N (Jul 13),24.70,25.15,24.56,25.10,24.90,0.50,194,0,152
"""

    rows = parse_contract_csv(
        csv_text,
        source="archive",
        source_file="CFE_N13_VX.csv",
        contract_expiry=date(2013, 7, 1),
    )

    assert [row["trade_date"] for row in rows] == [
        date(2012, 10, 22),
        date(2012, 10, 23),
    ]
    assert [row["settle"] for row in rows] == [24.4, 24.9]
    assert [row["contract_label"] for row in rows] == ["N (Jul 13)", "N (Jul 13)"]


def test_build_generic_contract_frame_supports_staggered_starts_and_roll_costs() -> (
    None
):
    trade_dates = [date(2024, 1, day) for day in range(2, 17)]

    def make_contract_rows(
        code: str,
        expiry: date,
        start_index: int,
        end_index: int,
        settle_base: float,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for offset, trade_date in enumerate(trade_dates[start_index:end_index]):
            settle = settle_base + offset
            rows.append(
                {
                    "trade_date": trade_date,
                    "contract_expiry": expiry,
                    "contract_code": code,
                    "contract_label": code,
                    "source": "synthetic",
                    "source_file": f"{code}.csv",
                    "open": settle,
                    "high": settle,
                    "low": settle,
                    "close": settle,
                    "settle": settle,
                    "change": None if offset == 0 else 1.0,
                    "total_volume": 1_000,
                    "efp": 0,
                    "open_interest": 2_000,
                }
            )
        return rows

    contract_rows: list[dict[str, object]] = []
    contract_rows.extend(make_contract_rows("C1", trade_dates[6], 0, 7, 100.0))
    contract_rows.extend(make_contract_rows("C2", trade_dates[7], 0, 8, 120.0))
    contract_rows.extend(make_contract_rows("C3", trade_dates[8], 0, 9, 140.0))
    contract_rows.extend(make_contract_rows("C4", trade_dates[9], 0, 10, 160.0))
    contract_rows.extend(make_contract_rows("C5", trade_dates[10], 0, 11, 180.0))
    contract_rows.extend(make_contract_rows("C6", trade_dates[11], 0, 12, 200.0))
    contract_rows.extend(make_contract_rows("C7", trade_dates[12], 1, 13, 220.0))
    contract_rows.extend(make_contract_rows("C8", trade_dates[13], 2, 14, 240.0))

    generic_frame = build_generic_contract_frame(pd.DataFrame(contract_rows))

    day_one = generic_frame.loc[
        generic_frame["trade_date"] == trade_dates[0]
    ].sort_values("ux_rank")
    assert day_one["ux_symbol"].tolist() == [
        "UX1",
        "UX2",
        "UX3",
        "UX4",
        "UX5",
        "UX6",
    ]

    ux7_start = (
        generic_frame.loc[generic_frame["ux_symbol"] == "UX7"]
        .sort_values("trade_date")
        .iloc[0]
        .to_dict()
    )
    assert ux7_start["trade_date"] == trade_dates[1]
    assert ux7_start["contract_code"] == "C7"
    assert pd.isna(ux7_start["gross_return"])
    assert pd.isna(ux7_start["net_return"])
    assert ux7_start["gross_index"] == 1.0
    assert ux7_start["net_index"] == 1.0

    ux1_last_old_contract_day = (
        generic_frame.loc[
            (generic_frame["ux_symbol"] == "UX1")
            & (generic_frame["trade_date"] == trade_dates[2])
        ]
        .iloc[0]
        .to_dict()
    )
    assert ux1_last_old_contract_day["contract_code"] == "C1"
    assert ux1_last_old_contract_day["rolled_today"] is False

    ux1_roll_day = (
        generic_frame.loc[
            (generic_frame["ux_symbol"] == "UX1")
            & (generic_frame["trade_date"] == trade_dates[3])
        ]
        .iloc[0]
        .to_dict()
    )
    expected_roll_day_return = 103.0 / 102.0 - 1.0

    assert ux1_roll_day["contract_code"] == "C2"
    assert ux1_roll_day["previous_contract_code"] == "C1"
    assert ux1_roll_day["rolled_today"] is True
    assert ux1_roll_day["transaction_cost"] == 0.0002
    assert ux1_roll_day["gross_return"] == expected_roll_day_return
    assert ux1_roll_day["net_return"] == expected_roll_day_return - 0.0002

    ux1_next_roll_day = (
        generic_frame.loc[
            (generic_frame["ux_symbol"] == "UX1")
            & (generic_frame["trade_date"] == trade_dates[4])
        ]
        .iloc[0]
        .to_dict()
    )
    expected_post_roll_return = 124.0 / 123.0 - 1.0

    assert ux1_next_roll_day["contract_code"] == "C3"
    assert ux1_next_roll_day["previous_contract_code"] == "C2"
    assert ux1_next_roll_day["rolled_today"] is True
    assert ux1_next_roll_day["gross_return"] == expected_post_roll_return


def test_parse_product_daily_csv_skips_disclaimer_and_selects_vx_columns() -> None:
    csv_text = """Disclaimer row,,,,
Date,VOLATILITY INDEX VOLUME,VOLATILITY INDEX OI,Other
3/26/2004,461,368,x
3/29/2004,117,349,y
"""

    rows = parse_product_daily_csv(csv_text)

    assert rows == [
        {"trade_date": date(2004, 3, 26), "vx_volume": 461, "vx_open_interest": 368},
        {"trade_date": date(2004, 3, 29), "vx_volume": 117, "vx_open_interest": 349},
    ]


def test_run_pipeline_writes_raw_and_clean_monthly_outputs(tmp_path: Path) -> None:
    config = SourceConfig(
        archive_page_url="https://example.test/archive-page",
        current_index_url="https://example.test/current-index",
        product_daily_url="https://example.test/cfevoloi.csv",
        archive_cdn_base="https://example.test/archive",
        current_cdn_base="https://example.test",
    )

    archive_page = '<script src="/_next/static/chunks/app/settlement-archive/page-test.js"></script>'
    archive_chunk = (
        'let V={"VX - Cboe S&P 500 Volatility Index (VIX) Futures Price and Volume Detail":'
        '{2013:[{path:"CFE_F13_VX.csv"}]}}'
    )
    current_index = {
        "2013": [
            {
                "product_display": "VX+VXT/F3",
                "expire_date": "2013-01-16",
                "futures_root": "VX",
                "duration_type": "M",
                "path": "data/us/futures/market_statistics/historical_data/VX/VX_2013-01-16.csv",
            }
        ],
        "2014": [
            {
                "product_display": "VX+VXT/F4",
                "expire_date": "2014-02-19",
                "futures_root": "VX",
                "duration_type": "M",
                "path": "data/us/futures/market_statistics/historical_data/VX/VX_2014-02-19.csv",
            },
            {
                "product_display": "VX+VXT32/Q5",
                "expire_date": "2015-08-05",
                "futures_root": "VX",
                "duration_type": "W",
                "path": "data/us/futures/market_statistics/historical_data/VX/VX_2015-08-05.csv",
            },
        ],
    }
    archive_csv = """Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest
01/15/2013,F (Jan 13),14.25,14.40,13.90,14.20,14.20,0.10,43022,74,57219
01/16/2013,F (Jan 13),0.00,14.25,14.40,0.00,13.69,-0.51,0,0,50051
"""
    current_csv = """Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest
2014-01-21,G (Feb 2014),13.40,13.45,12.83,13.23,13.25,-0.20,65202,32,44850
2014-01-22,G (Feb 2014),0.00,0.00,0.00,0.00,12.36,-0.89,0,0,40642
"""
    product_daily_csv = """Disclaimer row,,,,
Date,VOLATILITY INDEX VOLUME,VOLATILITY INDEX OI,Other
3/26/2004,461,368,x
"""

    responses = {
        "https://example.test/archive-page": httpx.Response(200, text=archive_page),
        "https://example.test/_next/static/chunks/app/settlement-archive/page-test.js": httpx.Response(
            200, text=archive_chunk
        ),
        "https://example.test/current-index": httpx.Response(200, json=current_index),
        "https://example.test/cfevoloi.csv": httpx.Response(
            200, text=product_daily_csv
        ),
        "https://example.test/archive/CFE_F13_VX.csv": httpx.Response(
            200, text=archive_csv
        ),
        "https://example.test/data/us/futures/market_statistics/historical_data/VX/VX_2014-02-19.csv": httpx.Response(
            200, text=current_csv
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses.get(str(request.url))
        if response is None:
            raise AssertionError(f"Unexpected request: {request.url}")
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = run_pipeline(
        data_dir=tmp_path / "data", refresh=False, client=client, config=config
    )

    assert result.contract_files == 2
    assert result.contract_rows == 4
    assert result.product_rows == 1

    archive_raw = (
        tmp_path / "data" / "raw" / "vix_futures" / "archive" / "CFE_F13_VX.csv"
    )
    current_raw = (
        tmp_path / "data" / "raw" / "vix_futures" / "current" / "VX_2014-02-19.csv"
    )
    skipped_current_2013 = (
        tmp_path / "data" / "raw" / "vix_futures" / "current" / "VX_2013-01-16.csv"
    )
    contract_clean = (
        tmp_path / "data" / "clean" / "vix_futures" / "monthly_contracts.parquet"
    )
    product_clean = (
        tmp_path / "data" / "clean" / "vix_futures" / "product_daily.parquet"
    )
    generic_clean = (
        tmp_path / "data" / "clean" / "vix_futures" / "generic_contracts.parquet"
    )

    assert archive_raw.exists()
    assert current_raw.exists()
    assert not skipped_current_2013.exists()
    assert contract_clean.exists()
    assert product_clean.exists()
    assert generic_clean.exists()

    contract_frame = pd.read_parquet(contract_clean)
    product_frame = pd.read_parquet(product_clean)

    assert parquet_date_list(contract_frame["contract_expiry"]) == [
        date(2013, 1, 16),
        date(2013, 1, 16),
        date(2014, 2, 19),
        date(2014, 2, 19),
    ]
    assert contract_frame["source"].tolist() == [
        "archive",
        "archive",
        "current",
        "current",
    ]
    product_records = product_frame.assign(
        trade_date=product_frame["trade_date"].map(
            lambda value: pd.Timestamp(value).date()
        )
    ).to_dict(orient="records")
    assert product_records == [
        {"trade_date": date(2004, 3, 26), "vx_volume": 461, "vx_open_interest": 368}
    ]
