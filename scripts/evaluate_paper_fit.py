from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from squid_replication.signals import (
    DEFAULT_BUCKET_ORDER,
    DEFAULT_DISLOCATION_TIE_POLICY,
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_UX_SYMBOLS,
    STABLE_DISLOCATION_TIE_POLICY,
    build_available_contract_frame,
    build_base_program_weights,
    build_daily_signal_frame,
    build_generic_return_frame,
    build_refined_program_weights,
    build_return_frame_from_levels,
    build_strategy_transaction_costs,
    build_spvxtstr_program_weights,
    build_term_structure,
    build_turnover_series,
    build_weekly_signal_frame,
    combine_weighted_returns,
    load_spot_vix_history,
    load_spvxtstr_history,
    summarize_conditional_returns,
    summarize_performance,
    summarize_bucket_returns,
)


PAPER_BUCKET_COUNTS = pd.Series(
    [2761, 887, 1311],
    index=DEFAULT_BUCKET_ORDER,
    dtype=float,
)
PAPER_SAME_DAY_RETURNS = pd.DataFrame(
    {
        "ES": [0.108, 0.085, -0.134],
        "SPVXTSTR": [0.040, 0.131, 0.003],
    },
    index=DEFAULT_BUCKET_ORDER,
)
PAPER_NEXT_DAY_RETURNS = pd.DataFrame(
    {
        "ES": [0.0360, 0.0060, 0.0729],
        "SPVXTSTR": [-0.0070, 0.0986, 0.1288],
    },
    index=DEFAULT_BUCKET_ORDER,
)
PAPER_HIGH_BUCKET_STATS = pd.DataFrame(
    {
        "mean": [0.0729, 0.1288],
        "median": [0.1569, 0.1362],
    },
    index=["ES", "SPVXTSTR"],
)
PAPER_TABLE_3_TARGETS = {
    "ES": {
        "annualized_return": 0.1077,
        "annualized_volatility": 0.1726,
        "max_drawdown": 0.5519,
        "sharpe_ratio": 0.54,
        "sortino_ratio": 0.44,
    },
    "Squid": {
        "annualized_return": 0.1584,
        "annualized_volatility": 0.1349,
        "max_drawdown": 0.2409,
        "sharpe_ratio": 1.06,
        "sortino_ratio": 1.18,
        "turnover": 0.0446,
    },
    "Cuttlefish": {
        "annualized_return": 0.1759,
        "annualized_volatility": 0.1342,
        "max_drawdown": 0.3010,
        "sharpe_ratio": 1.20,
        "sortino_ratio": 1.42,
        "turnover": 0.1545,
    },
}
PAPER_TABLE_4_TARGETS = {
    "Giant Squid": {
        "annualized_return": 0.2101,
        "annualized_volatility": 0.1534,
        "max_drawdown": 0.2414,
        "sharpe_ratio": 1.27,
        "sortino_ratio": 1.69,
        "turnover": 0.0823,
    },
    "Jumbo Squid": {
        "annualized_return": 0.1940,
        "annualized_volatility": 0.1372,
        "max_drawdown": 0.2409,
        "sharpe_ratio": 1.30,
        "sortino_ratio": 1.60,
        "turnover": 0.0608,
    },
    "Colossal Squid": {
        "annualized_return": 0.2109,
        "annualized_volatility": 0.1490,
        "max_drawdown": 0.2971,
        "sharpe_ratio": 1.31,
        "sortino_ratio": 1.73,
        "turnover": 0.2659,
    },
}


@dataclass(frozen=True)
class CandidateConfig:
    equity_proxy: str
    signal_price_field: str
    signal_selection_mode: str
    signal_tie_policy: str
    signal_forward_fill: bool
    vx_pnl_mode: str
    vx_price_field: str
    vx_selection_mode: str
    slope_threshold: float = -0.10
    vix_threshold: float = 30.0


def fetch_proxy_close(ticker: str, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    history = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if history.empty:
        raise ValueError(f"No history returned for {ticker}.")
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    close = history["Close"].rename(ticker)
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def build_contract_curve(
    generic_frame: pd.DataFrame,
    *,
    price_field: str,
    selection_mode: str,
    symbols: tuple[str, ...] = DEFAULT_UX_SYMBOLS,
    start: pd.Timestamp = DEFAULT_START,
    end: pd.Timestamp = DEFAULT_END,
    forward_fill: bool = False,
) -> pd.DataFrame:
    if selection_mode == "calendar_rank":
        curve = build_term_structure(
            generic_frame,
            price_field=price_field,
            symbols=symbols,
            start=start,
            end=end,
            forward_fill=forward_fill,
        )
    elif selection_mode == "available_ordinal":
        positions = {symbol: rank for rank, symbol in enumerate(symbols, start=1)}
        curve = build_available_contract_frame(
            generic_frame,
            value_field=price_field,
            contract_positions=positions,
            start=start,
            end=end,
        )
        if forward_fill:
            curve = curve.ffill()
    else:
        raise ValueError(f"Unsupported selection mode: {selection_mode!r}")
    return curve


def build_vx_return_frame(generic_frame: pd.DataFrame, candidate: CandidateConfig) -> pd.DataFrame:
    if candidate.vx_selection_mode == "calendar_rank":
        if candidate.vx_pnl_mode == "generic_gross_return":
            return build_generic_return_frame(
                generic_frame,
                return_field="gross_return",
                start=DEFAULT_START,
                end=DEFAULT_END,
                symbols=["UX1", "UX3"],
            )
        level_frame = build_term_structure(
            generic_frame,
            price_field=candidate.vx_price_field,
            start=DEFAULT_START,
            end=DEFAULT_END,
            symbols=["UX1", "UX3"],
        )
    elif candidate.vx_selection_mode == "available_ordinal":
        value_field = "gross_return" if candidate.vx_pnl_mode == "generic_gross_return" else candidate.vx_price_field
        level_frame = build_available_contract_frame(
            generic_frame,
            value_field=value_field,
            contract_positions={"UX1": 1, "UX3": 3},
            start=DEFAULT_START,
            end=DEFAULT_END,
        )
        if candidate.vx_pnl_mode == "generic_gross_return":
            return level_frame
    else:
        raise ValueError(f"Unsupported selection mode: {candidate.vx_selection_mode!r}")

    return build_return_frame_from_levels(level_frame)


def build_vx_roll_cost_frame(generic_frame: pd.DataFrame, candidate: CandidateConfig) -> pd.DataFrame:
    if candidate.vx_selection_mode == "calendar_rank":
        return build_generic_return_frame(
            generic_frame,
            return_field="transaction_cost",
            start=DEFAULT_START,
            end=DEFAULT_END,
            symbols=["UX1", "UX3"],
        )
    if candidate.vx_selection_mode == "available_ordinal":
        return build_available_contract_frame(
            generic_frame,
            value_field="transaction_cost",
            contract_positions={"UX1": 1, "UX3": 3},
            start=DEFAULT_START,
            end=DEFAULT_END,
        )
    raise ValueError(f"Unsupported selection mode: {candidate.vx_selection_mode!r}")


def normalized_error(actual: float, target: float) -> float:
    scale = max(abs(target), 0.01)
    return abs(actual - target) / scale


def table_fit_score(summary_table: pd.DataFrame, targets: dict[str, dict[str, float]]) -> float:
    total = 0.0
    for program, target_metrics in targets.items():
        for metric, target in target_metrics.items():
            actual_value = summary_table.loc[metric, program]
            if pd.isna(actual_value):
                total += 10.0
                continue
            actual = float(actual_value)
            if metric == "max_drawdown":
                actual = abs(actual)
            total += normalized_error(actual, target)
    return total


def evaluate_candidate(
    generic_frame: pd.DataFrame,
    proxy_returns: pd.DataFrame,
    spvxtstr: pd.DataFrame,
    vix_close: pd.Series,
    candidate: CandidateConfig,
) -> dict[str, float | str | bool]:
    signal_curve = build_contract_curve(
        generic_frame,
        price_field=candidate.signal_price_field,
        selection_mode=candidate.signal_selection_mode,
        forward_fill=candidate.signal_forward_fill,
    )
    daily_signal = build_daily_signal_frame(
        signal_curve,
        vix_close=vix_close,
        tie_policy=candidate.signal_tie_policy,
    )
    weekly_signal = build_weekly_signal_frame(
        signal_curve,
        vix_close=vix_close,
        tie_policy=candidate.signal_tie_policy,
    )

    vx_returns = build_vx_return_frame(generic_frame, candidate)
    vx_roll_costs = build_vx_roll_cost_frame(generic_frame, candidate)
    equity_series = proxy_returns[[candidate.equity_proxy]].rename(columns={candidate.equity_proxy: "ES"})
    spv_returns = spvxtstr[["spvxtstr_return"]].rename(columns={"spvxtstr_return": "SPVXTSTR"})
    analysis_returns = pd.concat([equity_series, spv_returns], axis=1, sort=False)
    strategy_returns = pd.concat([equity_series, vx_returns], axis=1, sort=False)

    same_day = summarize_bucket_returns(analysis_returns, daily_signal["dislocation_bucket_raw"])
    next_day = summarize_bucket_returns(
        analysis_returns,
        daily_signal["dislocation_bucket_raw"],
        future_periods=1,
    )
    high_mask = daily_signal["dislocation_bucket_raw"].eq("High Dislocation (5-7)")
    high_stats = summarize_conditional_returns(analysis_returns.shift(-1), high_mask)

    s0_weights = build_spvxtstr_program_weights(daily_signal, name="S0")
    daily_base_weights = build_base_program_weights(daily_signal, name="Cuttlefish")
    weekly_base_weights = build_base_program_weights(weekly_signal, name="Squid")
    daily_refined_weights = build_refined_program_weights(
        daily_signal,
        name="Colossal Squid",
        slope_threshold=candidate.slope_threshold,
        vix_threshold=candidate.vix_threshold,
        allow_short_es=True,
    )
    weekly_refined_weights = build_refined_program_weights(
        weekly_signal,
        name="Giant Squid",
        slope_threshold=candidate.slope_threshold,
        vix_threshold=candidate.vix_threshold,
        allow_short_es=True,
    )
    weekly_long_only_weights = build_refined_program_weights(
        weekly_signal,
        name="Jumbo Squid",
        slope_threshold=candidate.slope_threshold,
        vix_threshold=candidate.vix_threshold,
        allow_short_es=False,
    )

    s0_costs = build_strategy_transaction_costs(
        s0_weights,
        weight_columns=["weight_es", "weight_spvxtstr"],
    )
    daily_base_costs = build_strategy_transaction_costs(
        daily_base_weights,
        weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
        roll_costs=vx_roll_costs,
        roll_weight_map={"UX1": "weight_ux1", "UX3": "weight_ux3"},
    )
    weekly_base_costs = build_strategy_transaction_costs(
        weekly_base_weights,
        weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
        roll_costs=vx_roll_costs,
        roll_weight_map={"UX1": "weight_ux1", "UX3": "weight_ux3"},
    )
    weekly_refined_costs = build_strategy_transaction_costs(
        weekly_refined_weights,
        weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
        roll_costs=vx_roll_costs,
        roll_weight_map={"UX1": "weight_ux1", "UX3": "weight_ux3"},
    )
    weekly_long_only_costs = build_strategy_transaction_costs(
        weekly_long_only_weights,
        weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
        roll_costs=vx_roll_costs,
        roll_weight_map={"UX1": "weight_ux1", "UX3": "weight_ux3"},
    )
    daily_refined_costs = build_strategy_transaction_costs(
        daily_refined_weights,
        weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
        roll_costs=vx_roll_costs,
        roll_weight_map={"UX1": "weight_ux1", "UX3": "weight_ux3"},
    )

    s0_asset_returns = analysis_returns
    s0_returns = combine_weighted_returns(
        s0_asset_returns,
        s0_weights,
        asset_weight_map={"ES": "weight_es", "SPVXTSTR": "weight_spvxtstr"},
        name="S0",
        transaction_costs=s0_costs,
    )
    cuttlefish_returns = combine_weighted_returns(
        strategy_returns,
        daily_base_weights,
        asset_weight_map={"ES": "weight_es", "UX1": "weight_ux1", "UX3": "weight_ux3"},
        name="Cuttlefish",
        transaction_costs=daily_base_costs,
    )
    squid_returns = combine_weighted_returns(
        strategy_returns,
        weekly_base_weights,
        asset_weight_map={"ES": "weight_es", "UX1": "weight_ux1", "UX3": "weight_ux3"},
        name="Squid",
        transaction_costs=weekly_base_costs,
    )
    giant_returns = combine_weighted_returns(
        strategy_returns,
        weekly_refined_weights,
        asset_weight_map={"ES": "weight_es", "UX1": "weight_ux1", "UX3": "weight_ux3"},
        name="Giant Squid",
        transaction_costs=weekly_refined_costs,
    )
    jumbo_returns = combine_weighted_returns(
        strategy_returns,
        weekly_long_only_weights,
        asset_weight_map={"ES": "weight_es", "UX1": "weight_ux1", "UX3": "weight_ux3"},
        name="Jumbo Squid",
        transaction_costs=weekly_long_only_costs,
    )
    colossal_returns = combine_weighted_returns(
        strategy_returns,
        daily_refined_weights,
        asset_weight_map={"ES": "weight_es", "UX1": "weight_ux1", "UX3": "weight_ux3"},
        name="Colossal Squid",
        transaction_costs=daily_refined_costs,
    )

    base_common = pd.concat([equity_series, cuttlefish_returns, squid_returns], axis=1, sort=False).dropna(how="any")
    refined_common = pd.concat(
        [equity_series, giant_returns, jumbo_returns, colossal_returns],
        axis=1,
        sort=False,
    ).dropna(how="any")

    base_table = pd.DataFrame(
        {
            "ES": summarize_performance(base_common["ES"]),
            "Squid": summarize_performance(
                base_common["Squid"],
                turnover=build_turnover_series(
                    weekly_base_weights,
                    weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
                    one_way=False,
                ).reindex(base_common.index),
            ),
            "Cuttlefish": summarize_performance(
                base_common["Cuttlefish"],
                turnover=build_turnover_series(
                    daily_base_weights,
                    weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
                    one_way=False,
                ).reindex(base_common.index),
            ),
        }
    )
    refined_table = pd.DataFrame(
        {
            "Giant Squid": summarize_performance(
                refined_common["Giant Squid"],
                turnover=build_turnover_series(
                    weekly_refined_weights,
                    weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
                    one_way=False,
                ).reindex(refined_common.index),
            ),
            "Jumbo Squid": summarize_performance(
                refined_common["Jumbo Squid"],
                turnover=build_turnover_series(
                    weekly_long_only_weights,
                    weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
                    one_way=False,
                ).reindex(refined_common.index),
            ),
            "Colossal Squid": summarize_performance(
                refined_common["Colossal Squid"],
                turnover=build_turnover_series(
                    daily_refined_weights,
                    weight_columns=["weight_es", "weight_ux1", "weight_ux3"],
                    one_way=False,
                ).reindex(refined_common.index),
            ),
        }
    )

    count_error = (same_day["count"].astype(float) - PAPER_BUCKET_COUNTS).abs().mean() / PAPER_BUCKET_COUNTS.sum()
    same_day_error = (
        (100 * same_day[["ES", "SPVXTSTR"]] - PAPER_SAME_DAY_RETURNS).abs().stack().mean()
    )
    next_day_error = (
        (100 * next_day[["ES", "SPVXTSTR"]] - PAPER_NEXT_DAY_RETURNS).abs().stack().mean()
    )
    high_stats_error = (
        (100 * high_stats.loc[["ES", "SPVXTSTR"], ["mean", "median"]] - PAPER_HIGH_BUCKET_STATS)
        .abs()
        .stack()
        .mean()
    )
    base_fit = table_fit_score(base_table, PAPER_TABLE_3_TARGETS)
    refined_fit = table_fit_score(refined_table, PAPER_TABLE_4_TARGETS)

    signal_start = daily_signal["es_weight_signal"].dropna().index.min()
    vx_start = vx_returns.dropna(how="any").index.min()
    strategy_start = cuttlefish_returns.dropna().index.min()

    results = asdict(candidate)
    results.update(
        {
            "signal_start": None if pd.isna(signal_start) else signal_start.date().isoformat(),
            "vx_start": None if pd.isna(vx_start) else vx_start.date().isoformat(),
            "strategy_start": None if pd.isna(strategy_start) else strategy_start.date().isoformat(),
            "count_mae_ratio": float(count_error),
            "same_day_bucket_mae_pct": float(same_day_error),
            "next_day_bucket_mae_pct": float(next_day_error),
            "high_bucket_stats_mae_pct": float(high_stats_error),
            "base_table_fit_score": float(base_fit),
            "refined_table_fit_score": float(refined_fit),
            "total_fit_score": float(count_error + same_day_error + next_day_error + high_stats_error + base_fit + refined_fit),
            "same_day_perfect_count": int(
                pd.Series([same_day.loc[DEFAULT_BUCKET_ORDER[0], "count"]])
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
                .iloc[0]
            ),
            "same_day_low_count": int(
                pd.Series([same_day.loc[DEFAULT_BUCKET_ORDER[1], "count"]])
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
                .iloc[0]
            ),
            "same_day_high_count": int(
                pd.Series([same_day.loc[DEFAULT_BUCKET_ORDER[2], "count"]])
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
                .iloc[0]
            ),
        }
    )
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    generic_path = root / "data" / "clean" / "vix_futures" / "generic_contracts.parquet"
    spvxtstr_path = root / "data" / "external" / "spvxtstr_normalized.csv"

    generic_frame = pd.read_parquet(generic_path)
    spvxtstr = load_spvxtstr_history(spvxtstr_path).loc[DEFAULT_START:DEFAULT_END]
    vix_close = load_spot_vix_history()["vix_close"].loc[DEFAULT_START:DEFAULT_END]

    proxy_close = pd.concat(
        [
            fetch_proxy_close("ES=F", start=DEFAULT_START, end=DEFAULT_END),
            fetch_proxy_close("SPY", start=DEFAULT_START, end=DEFAULT_END),
            fetch_proxy_close("^GSPC", start=DEFAULT_START, end=DEFAULT_END),
        ],
        axis=1,
        sort=False,
    )
    proxy_close = proxy_close.reindex(pd.to_datetime(generic_frame["trade_date"]).sort_values().unique()).ffill()
    proxy_returns = proxy_close.pct_change()

    candidates: list[CandidateConfig] = []
    for equity_proxy in ("ES=F", "SPY", "^GSPC"):
        for signal_price_field in ("settle_expiry_roll", "close_expiry_roll", "settle", "close"):
            for signal_selection_mode in ("calendar_rank", "available_ordinal"):
                for signal_tie_policy in (
                    DEFAULT_DISLOCATION_TIE_POLICY,
                    STABLE_DISLOCATION_TIE_POLICY,
                ):
                    for signal_forward_fill in (False, True):
                        for vx_pnl_mode, vx_price_field in (
                            ("generic_gross_return", "settle_expiry_roll"),
                            ("level_returns", "settle_expiry_roll"),
                            ("level_returns", "close_expiry_roll"),
                        ):
                            for vx_selection_mode in (
                                "calendar_rank",
                                "available_ordinal",
                            ):
                                candidates.append(
                                    CandidateConfig(
                                        equity_proxy=equity_proxy,
                                        signal_price_field=signal_price_field,
                                        signal_selection_mode=signal_selection_mode,
                                        signal_tie_policy=signal_tie_policy,
                                        signal_forward_fill=signal_forward_fill,
                                        vx_pnl_mode=vx_pnl_mode,
                                        vx_price_field=vx_price_field,
                                        vx_selection_mode=vx_selection_mode,
                                    )
                                )

    rows = [evaluate_candidate(generic_frame, proxy_returns, spvxtstr, vix_close, candidate) for candidate in candidates]
    results = pd.DataFrame(rows).sort_values(
        ["total_fit_score", "base_table_fit_score", "refined_table_fit_score"]
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(results.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
