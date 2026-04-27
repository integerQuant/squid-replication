from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from squid_replication.signals import (
    DEFAULT_BUCKET_ORDER,
    build_available_contract_frame,
    build_base_program_weights,
    build_drawdown_series,
    build_daily_signal_frame,
    build_generic_return_frame,
    build_growth_index,
    build_refined_program_weights,
    build_return_frame_from_levels,
    build_strategy_transaction_costs,
    build_spvxtstr_program_weights,
    build_term_structure,
    build_turnover_series,
    build_weekly_signal_frame,
    combine_weighted_returns,
    count_curve_dislocations,
    extract_extreme_moves,
    lag_weekly_average,
    load_spvxtstr_history,
    load_spot_vix_history,
    normalize_decimal_text,
    summarize_performance,
    summarize_bucket_returns,
    summarize_conditional_returns,
)


def make_curve(values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.to_datetime(
            [f"2024-01-{day:02d}" for day in range(2, 2 + len(values))]
        ),
        columns=[f"UX{rank}" for rank in range(1, 8)],
    )


def test_build_term_structure_pivots_and_forward_fills_missing_contracts() -> None:
    generic_frame = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "ux_symbol": "UX1",
                "settle_expiry_roll": 12.0,
                "settle": 11.9,
            },
            {
                "trade_date": "2024-01-02",
                "ux_symbol": "UX2",
                "settle_expiry_roll": 13.0,
                "settle": 12.9,
            },
            {
                "trade_date": "2024-01-02",
                "ux_symbol": "UX3",
                "settle_expiry_roll": 14.0,
                "settle": 13.9,
            },
            {
                "trade_date": "2024-01-03",
                "ux_symbol": "UX1",
                "settle_expiry_roll": 12.5,
                "settle": 12.4,
            },
            {
                "trade_date": "2024-01-03",
                "ux_symbol": "UX3",
                "settle_expiry_roll": 14.5,
                "settle": 14.4,
            },
        ]
    )

    curve = build_term_structure(
        generic_frame,
        price_field="settle_expiry_roll",
        symbols=["UX1", "UX2", "UX3"],
        forward_fill=True,
    )

    assert list(curve.columns) == ["UX1", "UX2", "UX3"]
    assert curve.loc[pd.Timestamp("2024-01-03"), "UX2"] == 13.0


def test_count_curve_dislocations_matches_contango_backwardation_and_ties() -> None:
    curve = make_curve(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1],
            [2, 1, 3, 4, 5, 6, 7],
            [5, 5, 5, 5, 5, 5, 5],
        ]
    )

    dislocations = count_curve_dislocations(curve)

    assert dislocations.tolist() == [0, 6, 2, 6]


def test_count_curve_dislocations_supports_stable_tie_handling() -> None:
    curve = make_curve(
        [
            [1, 2, 3, 4, 5, 5, 7],
            [5, 5, 5, 5, 5, 5, 5],
            [2, 1, 3, 4, 5, 6, 7],
        ]
    )

    strict = count_curve_dislocations(curve)
    stable = count_curve_dislocations(curve, tie_policy="stable_order")

    assert strict.tolist() == [1, 6, 2]
    assert stable.tolist() == [0, 0, 2]


def test_build_daily_signal_frame_shifts_signal_one_day() -> None:
    signal_curve = make_curve(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1],
            [2, 1, 3, 4, 5, 6, 7],
        ]
    )
    vix_close = pd.Series(
        [18.0, 35.0, 22.0],
        index=signal_curve.index,
        name="vix_close",
    )

    signal_frame = build_daily_signal_frame(signal_curve, vix_close=vix_close)

    assert signal_frame.loc[pd.Timestamp("2024-01-02"), "dislocation_bucket_raw"] == (
        "Perfect Contango (0)"
    )
    assert signal_frame.loc[pd.Timestamp("2024-01-03"), "dislocation_bucket_raw"] == (
        "High Dislocation (5-7)"
    )
    assert signal_frame.loc[pd.Timestamp("2024-01-04"), "dislocation_bucket_raw"] == (
        "Low Dislocation (2-4)"
    )
    assert pd.isna(signal_frame.loc[pd.Timestamp("2024-01-02"), "es_weight_signal"])
    assert signal_frame.loc[pd.Timestamp("2024-01-03"), "es_weight_signal"] == pytest.approx(
        1.0
    )
    assert signal_frame.loc[pd.Timestamp("2024-01-04"), "es_weight_signal"] == pytest.approx(
        1.0 / 7.0
    )
    assert signal_frame.loc[pd.Timestamp("2024-01-04"), "slope_signal"] == pytest.approx(
        -6.0
    )
    assert signal_frame.loc[pd.Timestamp("2024-01-04"), "vix_close_signal"] == pytest.approx(
        35.0
    )


def test_build_daily_signal_frame_accepts_stable_tie_policy() -> None:
    signal_curve = make_curve(
        [
            [1, 2, 3, 4, 5, 5, 7],
            [2, 1, 3, 4, 5, 6, 7],
        ]
    )

    signal_frame = build_daily_signal_frame(signal_curve, tie_policy="stable_order")

    assert signal_frame.loc[pd.Timestamp("2024-01-02"), "dislocation_count_raw"] == 0
    assert signal_frame.loc[pd.Timestamp("2024-01-02"), "dislocation_bucket_raw"] == (
        "Perfect Contango (0)"
    )
    assert signal_frame.loc[pd.Timestamp("2024-01-03"), "es_weight_signal"] == pytest.approx(
        1.0
    )


def test_lag_weekly_average_uses_previous_completed_week() -> None:
    signal = pd.Series(
        [0.0, 2.0, 4.0, 6.0, 1.0, 3.0],
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
            ]
        ),
        name="signal",
    )

    lagged = lag_weekly_average(signal)

    assert lagged.iloc[:4].isna().all()
    assert lagged.iloc[4] == pytest.approx(3.0)
    assert lagged.iloc[5] == pytest.approx(3.0)


def test_build_weekly_signal_frame_uses_previous_week_average_dislocations() -> None:
    signal_curve = make_curve(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [2, 1, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 1, 3, 4, 5, 6, 7],
        ]
    )
    signal_curve.index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )

    signal_frame = build_weekly_signal_frame(signal_curve)

    # Week one dislocations are 0, 2, 6, 0 => mean 2.0.
    assert signal_frame.loc[pd.Timestamp("2024-01-08"), "weekly_dislocation_mean_signal"] == pytest.approx(
        2.0
    )
    assert signal_frame.loc[pd.Timestamp("2024-01-09"), "es_weight_signal"] == pytest.approx(
        5.0 / 7.0
    )


def test_build_base_program_weights_converts_es_signal_into_component_exposures() -> None:
    signal_frame = pd.DataFrame(
        {"es_weight_signal": [None, 1.0, 0.5]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    weights = build_base_program_weights(signal_frame, name="Cuttlefish")

    assert pd.isna(weights.loc[pd.Timestamp("2024-01-02"), "weight_es"])
    assert weights.loc[pd.Timestamp("2024-01-03"), "weight_va"] == pytest.approx(0.0)
    assert weights.loc[pd.Timestamp("2024-01-04"), "weight_ux1"] == pytest.approx(-0.25)
    assert weights.loc[pd.Timestamp("2024-01-04"), "weight_ux3"] == pytest.approx(0.5)
    assert weights.loc[pd.Timestamp("2024-01-04"), "gross_exposure"] == pytest.approx(
        1.25
    )


def test_build_refined_program_weights_supports_long_short_and_long_only_modes() -> None:
    signal_frame = pd.DataFrame(
        {
            "es_weight_signal": [0.5, 0.4],
            "slope_signal": [-0.05, 0.10],
            "vix_close_signal": [18.0, 35.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    long_short = build_refined_program_weights(
        signal_frame,
        name="Giant Squid",
        slope_threshold=-0.10,
        vix_threshold=30.0,
        allow_short_es=True,
    )
    long_only = build_refined_program_weights(
        signal_frame,
        name="Jumbo Squid",
        slope_threshold=-0.10,
        vix_threshold=30.0,
        allow_short_es=False,
    )

    assert long_short.loc[pd.Timestamp("2024-01-02"), "regime"] == "dampened"
    assert long_short.loc[pd.Timestamp("2024-01-02"), "weight_es"] == pytest.approx(
        0.75
    )
    assert long_short.loc[pd.Timestamp("2024-01-02"), "weight_va"] == pytest.approx(
        0.25
    )
    assert long_short.loc[pd.Timestamp("2024-01-03"), "regime"] == "double_vol"
    assert long_short.loc[pd.Timestamp("2024-01-03"), "weight_es"] == pytest.approx(
        -0.2
    )
    assert long_short.loc[pd.Timestamp("2024-01-03"), "weight_va"] == pytest.approx(
        1.2
    )
    assert long_only.loc[pd.Timestamp("2024-01-03"), "weight_es"] == pytest.approx(0.0)
    assert long_only.loc[pd.Timestamp("2024-01-03"), "weight_va"] == pytest.approx(1.0)


def test_load_spot_vix_history_normalizes_cboe_csv_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "VIX_History.csv"
    csv_path.write_text(
        "DATE,OPEN,HIGH,LOW,CLOSE\n"
        "01/02/2024,13.0,14.0,12.5,13.5\n",
        encoding="utf-8",
    )

    history = load_spot_vix_history(csv_path)

    assert list(history.columns) == ["open", "high", "low", "vix_close"]
    assert history.index[0] == pd.Timestamp("2024-01-02")
    assert history.iloc[0]["vix_close"] == pytest.approx(13.5)


def test_load_spvxtstr_history_normalizes_bloomberg_style_export(tmp_path: Path) -> None:
    csv_path = tmp_path / "spvxtstr.csv"
    csv_path.write_text(
        "date;SPVXTSTR\n"
        "05/04/2006;95901,7\n"
        "06/04/2006;95911,6\n",
        encoding="utf-8",
    )

    history = load_spvxtstr_history(csv_path)

    assert list(history.columns) == [
        "spvxtstr",
        "spvxtstr_return",
        "spvxtstr_index",
    ]
    assert history.index[0] == pd.Timestamp("2006-04-05")
    assert history.iloc[0]["spvxtstr"] == pytest.approx(95901.7)
    assert pd.isna(history.iloc[0]["spvxtstr_return"])
    assert history.iloc[1]["spvxtstr_return"] == pytest.approx(
        95911.6 / 95901.7 - 1.0
    )
    assert history.iloc[0]["spvxtstr_index"] == pytest.approx(1.0)


def test_load_spvxtstr_history_accepts_clean_normalized_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "spvxtstr_normalized.csv"
    csv_path.write_text(
        "trade_date,spvxtstr,spvxtstr_return,spvxtstr_index\n"
        "2006-04-05,95901.7,,1.0\n"
        "2006-04-06,95911.6,0.000103,1.000103\n",
        encoding="utf-8",
    )

    history = load_spvxtstr_history(csv_path)

    assert history.index[0] == pd.Timestamp("2006-04-05")
    assert history.iloc[0]["spvxtstr"] == pytest.approx(95901.7)
    assert history.iloc[1]["spvxtstr_return"] == pytest.approx(
        95911.6 / 95901.7 - 1.0
    )


def test_normalize_decimal_text_handles_decimal_comma_and_decimal_dot() -> None:
    assert normalize_decimal_text("100887,1") == "100887.1"
    assert normalize_decimal_text("95,901.7") == "95901.7"
    assert normalize_decimal_text("95901.7") == "95901.7"


def test_summarize_bucket_returns_supports_same_day_and_t_plus_one_views() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    returns = pd.DataFrame(
        {
            "ES": [0.01, 0.02, -0.01, 0.03],
            "SPVXTSTR": [0.005, 0.01, 0.015, 0.02],
        },
        index=index,
    )
    buckets = pd.Series(
        [
            "Perfect Contango (0)",
            "Low Dislocation (2-4)",
            "High Dislocation (5-7)",
            "Perfect Contango (0)",
        ],
        index=index,
        name="dislocation_bucket_raw",
    )

    same_day = summarize_bucket_returns(returns, buckets)
    next_day = summarize_bucket_returns(returns, buckets, future_periods=1)

    assert same_day.loc["Perfect Contango (0)", "count"] == 2
    assert same_day.loc["Perfect Contango (0)", "ES"] == pytest.approx(0.02)
    assert next_day.loc["Perfect Contango (0)", "count"] == 1
    assert next_day.loc["Perfect Contango (0)", "ES"] == pytest.approx(0.02)
    assert next_day.loc["Low Dislocation (2-4)", "SPVXTSTR"] == pytest.approx(0.015)
    assert list(next_day.index) == list(DEFAULT_BUCKET_ORDER)


def test_summarize_conditional_returns_reports_mean_median_and_count() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    returns = pd.DataFrame(
        {
            "ES": [0.01, -0.02, 0.03],
            "SPVXTSTR": [0.005, 0.01, 0.015],
        },
        index=index,
    )
    mask = pd.Series([False, True, True], index=index)

    summary = summarize_conditional_returns(returns, mask)

    assert summary.loc["ES", "count"] == 2
    assert summary.loc["ES", "mean"] == pytest.approx(0.005)
    assert summary.loc["ES", "median"] == pytest.approx(0.005)
    assert summary.loc["SPVXTSTR", "mean"] == pytest.approx(0.0125)


def test_extract_extreme_moves_filters_by_absolute_threshold() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    returns = pd.Series([0.01, 0.04, -0.05, 0.02], index=index, name="ES")
    mask = pd.Series([False, True, True, True], index=index)

    extremes = extract_extreme_moves(returns, mask, threshold=0.03)

    assert extremes.index.tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    assert extremes["return"].tolist() == [0.04, -0.05]
    assert extremes["direction"].tolist() == ["positive", "negative"]


def test_build_generic_return_frame_pivots_selected_return_field() -> None:
    generic_frame = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "ux_symbol": "UX1", "net_return": 0.01},
            {"trade_date": "2024-01-02", "ux_symbol": "UX3", "net_return": 0.03},
            {"trade_date": "2024-01-03", "ux_symbol": "UX1", "net_return": 0.02},
        ]
    )

    returns = build_generic_return_frame(
        generic_frame,
        return_field="net_return",
        symbols=["UX1", "UX3"],
        start="2024-01-02",
        end="2024-01-03",
    )

    assert list(returns.columns) == ["UX1", "UX3"]
    assert returns.loc[pd.Timestamp("2024-01-02"), "UX1"] == pytest.approx(0.01)
    assert pd.isna(returns.loc[pd.Timestamp("2024-01-03"), "UX3"])


def test_build_available_contract_frame_selects_nth_available_contracts() -> None:
    generic_frame = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "ux_symbol": "UX1",
                "ux_rank": 1,
                "contract_expiry": "2024-01-17",
                "net_return": 0.01,
            },
            {
                "trade_date": "2024-01-02",
                "ux_symbol": "UX2",
                "ux_rank": 2,
                "contract_expiry": "2024-02-14",
                "net_return": 0.02,
            },
            {
                "trade_date": "2024-01-02",
                "ux_symbol": "UX4",
                "ux_rank": 4,
                "contract_expiry": "2024-04-17",
                "net_return": 0.04,
            },
            {
                "trade_date": "2024-01-03",
                "ux_symbol": "UX1",
                "ux_rank": 1,
                "contract_expiry": "2024-01-17",
                "net_return": 0.011,
            },
            {
                "trade_date": "2024-01-03",
                "ux_symbol": "UX3",
                "ux_rank": 3,
                "contract_expiry": "2024-03-20",
                "net_return": 0.03,
            },
            {
                "trade_date": "2024-01-03",
                "ux_symbol": "UX4",
                "ux_rank": 4,
                "contract_expiry": "2024-04-17",
                "net_return": 0.041,
            },
        ]
    )

    available = build_available_contract_frame(
        generic_frame,
        value_field="net_return",
        contract_positions={"UX1": 1, "UX3": 3},
        start="2024-01-02",
        end="2024-01-03",
    )

    assert list(available.columns) == ["UX1", "UX3"]
    assert available.loc[pd.Timestamp("2024-01-02"), "UX1"] == pytest.approx(0.01)
    assert available.loc[pd.Timestamp("2024-01-02"), "UX3"] == pytest.approx(0.04)
    assert available.loc[pd.Timestamp("2024-01-03"), "UX1"] == pytest.approx(0.011)
    assert available.loc[pd.Timestamp("2024-01-03"), "UX3"] == pytest.approx(0.041)


def test_build_return_frame_from_levels_respects_missing_levels() -> None:
    levels = pd.DataFrame(
        {
            "UX1": [10.0, 11.0, None, 12.0],
            "UX3": [20.0, 21.0, 22.0, 23.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )

    returns = build_return_frame_from_levels(levels)

    assert pd.isna(returns.iloc[0]["UX1"])
    assert returns.loc[pd.Timestamp("2024-01-03"), "UX1"] == pytest.approx(0.10)
    assert pd.isna(returns.loc[pd.Timestamp("2024-01-04"), "UX1"])
    assert pd.isna(returns.loc[pd.Timestamp("2024-01-05"), "UX1"])
    assert returns.loc[pd.Timestamp("2024-01-04"), "UX3"] == pytest.approx(22.0 / 21.0 - 1.0)


def test_build_spvxtstr_program_weights_creates_two_asset_mix() -> None:
    signal_frame = pd.DataFrame(
        {"es_weight_signal": [None, 0.75, 0.25]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    weights = build_spvxtstr_program_weights(signal_frame, name="S0")

    assert pd.isna(weights.loc[pd.Timestamp("2024-01-02"), "weight_es"])
    assert weights.loc[pd.Timestamp("2024-01-03"), "weight_spvxtstr"] == pytest.approx(
        0.25
    )
    assert weights.loc[pd.Timestamp("2024-01-04"), "gross_exposure"] == pytest.approx(
        1.0
    )


def test_combine_weighted_returns_multiplies_assets_by_matching_weight_columns() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    asset_returns = pd.DataFrame(
        {"ES": [0.01, 0.02], "UX1": [-0.03, 0.01], "UX3": [0.04, 0.03]},
        index=index,
    )
    weights = pd.DataFrame(
        {
            "weight_es": [0.5, 0.25],
            "weight_ux1": [-0.25, -0.375],
            "weight_ux3": [0.5, 0.75],
        },
        index=index,
    )

    strategy_returns = combine_weighted_returns(
        asset_returns,
        weights,
        asset_weight_map={"ES": "weight_es", "UX1": "weight_ux1", "UX3": "weight_ux3"},
        name="strategy_return",
    )

    assert strategy_returns.name == "strategy_return"
    assert strategy_returns.iloc[0] == pytest.approx(0.5 * 0.01 + (-0.25) * (-0.03) + 0.5 * 0.04)
    assert strategy_returns.iloc[1] == pytest.approx(0.25 * 0.02 + (-0.375) * 0.01 + 0.75 * 0.03)


def test_build_growth_index_rebases_returns_to_one() -> None:
    returns = pd.Series(
        [None, 0.10, -0.05, 0.02],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )

    growth = build_growth_index(returns)

    assert pd.isna(growth.iloc[0])
    assert growth.iloc[1] == pytest.approx(1.10)
    assert growth.iloc[2] == pytest.approx(1.10 * 0.95)
    assert growth.iloc[3] == pytest.approx(1.10 * 0.95 * 1.02)


def test_build_drawdown_series_tracks_distance_from_running_peak() -> None:
    growth = pd.Series(
        [None, 1.0, 1.10, 1.00, 1.20],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]),
    )

    drawdown = build_drawdown_series(growth)

    assert pd.isna(drawdown.iloc[0])
    assert drawdown.iloc[1] == pytest.approx(0.0)
    assert drawdown.iloc[2] == pytest.approx(0.0)
    assert drawdown.iloc[3] == pytest.approx(1.00 / 1.10 - 1.0)
    assert drawdown.iloc[4] == pytest.approx(0.0)


def test_build_turnover_series_uses_one_way_absolute_weight_change() -> None:
    weights = pd.DataFrame(
        {
            "weight_es": [None, 0.5, 0.25],
            "weight_spvxtstr": [None, 0.5, 0.75],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    turnover = build_turnover_series(weights, weight_columns=["weight_es", "weight_spvxtstr"])

    assert pd.isna(turnover.iloc[0])
    assert pd.isna(turnover.iloc[1])
    assert turnover.iloc[2] == pytest.approx(0.25)


def test_build_turnover_series_can_report_full_traded_notional() -> None:
    weights = pd.DataFrame(
        {
            "weight_es": [None, 0.5, 0.25],
            "weight_spvxtstr": [None, 0.5, 0.75],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    turnover = build_turnover_series(
        weights,
        weight_columns=["weight_es", "weight_spvxtstr"],
        one_way=False,
    )

    assert pd.isna(turnover.iloc[0])
    assert pd.isna(turnover.iloc[1])
    assert turnover.iloc[2] == pytest.approx(0.5)


def test_build_strategy_transaction_costs_charges_absolute_rebalances_and_rolls() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    weights = pd.DataFrame(
        {
            "weight_es": [None, 0.5, 0.25],
            "weight_ux1": [None, -0.25, -0.375],
            "weight_ux3": [None, 0.5, 0.75],
        },
        index=index,
    )
    roll_costs = pd.DataFrame(
        {
            "UX1": [0.0, 0.0, 0.0002],
            "UX3": [0.0, 0.0002, 0.0],
        },
        index=index,
    )

    costs = build_strategy_transaction_costs(
        weights,
        weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
        roll_costs=roll_costs,
        roll_weight_map={"UX1": "weight_ux1", "UX3": "weight_ux3"},
        transaction_cost_bps=2.0,
    )

    expected_rebalance_cost = (0.25 + 0.125 + 0.25) * 0.0002
    expected_roll_cost = 0.375 * 0.0002
    assert pd.isna(costs.iloc[0])
    assert costs.iloc[1] == pytest.approx(0.0)
    assert costs.iloc[2] == pytest.approx(expected_rebalance_cost + expected_roll_cost)


def test_combine_weighted_returns_subtracts_transaction_costs() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    asset_returns = pd.DataFrame(
        {"ES": [0.01, 0.02], "UX1": [-0.03, 0.01], "UX3": [0.04, 0.03]},
        index=index,
    )
    weights = pd.DataFrame(
        {
            "weight_es": [0.5, 0.25],
            "weight_ux1": [-0.25, -0.375],
            "weight_ux3": [0.5, 0.75],
        },
        index=index,
    )
    transaction_costs = pd.Series([0.0, 0.0003], index=index)

    strategy_returns = combine_weighted_returns(
        asset_returns,
        weights,
        asset_weight_map={"ES": "weight_es", "UX1": "weight_ux1", "UX3": "weight_ux3"},
        transaction_costs=transaction_costs,
        name="strategy_return",
    )

    gross_return = 0.25 * 0.02 + (-0.375) * 0.01 + 0.75 * 0.03
    assert strategy_returns.iloc[1] == pytest.approx(gross_return - 0.0003)


def test_summarize_performance_reports_core_metrics_and_turnover() -> None:
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        name="strategy",
    )
    turnover = pd.Series(
        [None, 0.10, 0.20],
        index=returns.index,
        dtype=float,
    )

    summary = summarize_performance(returns, turnover=turnover, annualization=3)

    expected_ann_return = (1.10 * 0.95 * 1.02) - 1.0
    expected_ann_vol = returns.std(ddof=1) * (3**0.5)
    expected_sortino_denominator = ((returns.clip(upper=0) ** 2).mean()) ** 0.5 * (3**0.5)

    assert summary["annualized_return"] == pytest.approx(expected_ann_return)
    assert summary["annualized_volatility"] == pytest.approx(expected_ann_vol)
    assert summary["max_drawdown"] == pytest.approx(1.045 / 1.10 - 1.0)
    assert summary["sharpe_ratio"] == pytest.approx(expected_ann_return / expected_ann_vol)
    assert summary["sortino_ratio"] == pytest.approx(expected_ann_return / expected_sortino_denominator)
    assert summary["turnover"] == pytest.approx(0.15)


def test_summarize_performance_handles_nullable_single_observation() -> None:
    returns = pd.Series([0.01], index=pd.to_datetime(["2024-01-03"]), dtype="Float64")

    summary = summarize_performance(returns, annualization=1)

    assert summary["annualized_return"] == pytest.approx(0.01)
    assert pd.isna(summary["annualized_volatility"])
    assert pd.isna(summary["sharpe_ratio"])
    assert pd.isna(summary["sortino_ratio"])
