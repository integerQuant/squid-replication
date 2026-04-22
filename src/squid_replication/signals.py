from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_START = pd.Timestamp("2006-04-05")
DEFAULT_END = pd.Timestamp("2025-12-31")
DEFAULT_UX_SYMBOLS = tuple(f"UX{rank}" for rank in range(1, 8))

PERFECT_CONTANGO_BUCKET = "Perfect Contango (0)"
LOW_DISLOCATION_BUCKET = "Low Dislocation (2-4)"
HIGH_DISLOCATION_BUCKET = "High Dislocation (5-7)"
DEFAULT_BUCKET_ORDER = (
    PERFECT_CONTANGO_BUCKET,
    LOW_DISLOCATION_BUCKET,
    HIGH_DISLOCATION_BUCKET,
)

CBOE_VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def load_spot_vix_history(source: str | Path = CBOE_VIX_HISTORY_URL) -> pd.DataFrame:
    """Load the official daily spot VIX history from Cboe or a local CSV."""
    frame = pd.read_csv(source, parse_dates=["DATE"])
    frame = frame.rename(columns=str.lower).rename(
        columns={"date": "trade_date", "close": "vix_close"}
    )
    frame = frame[["trade_date", "open", "high", "low", "vix_close"]]
    return frame.sort_values("trade_date").set_index("trade_date")


def load_spvxtstr_history(source: str | Path) -> pd.DataFrame:
    """Load SPVXTSTR history from a flat CSV or Bloomberg-style semicolon export."""
    path = Path(source)
    header_line = path.open("r", encoding="utf-8-sig").readline().strip()
    separator = ";" if ";" in header_line else ","

    frame = pd.read_csv(path, sep=separator, dtype=str, encoding="utf-8-sig")
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    date_column = next(
        (column for column in frame.columns if column in {"trade_date", "date"}), None
    )
    value_column = next(
        (
            column
            for column in frame.columns
            if column in {"spvxtstr", "px_last", "close", "last_price"}
        ),
        None,
    )

    if date_column is None or value_column is None:
        raise ValueError(
            "SPVXTSTR CSV must include a date column and a value column such as "
            "'SPVXTSTR' or 'PX_LAST'."
        )

    normalized = pd.DataFrame()
    date_text = frame[date_column].astype(str).str.strip()
    iso_mask = date_text.str.match(r"^\d{4}-\d{2}-\d{2}$")
    parsed_iso = pd.to_datetime(date_text.where(iso_mask), errors="coerce")
    parsed_non_iso = pd.to_datetime(
        date_text.where(~iso_mask), dayfirst=True, errors="coerce"
    )
    normalized["trade_date"] = parsed_non_iso.where(~iso_mask, parsed_iso)
    numeric_text = frame[value_column].astype(str).map(normalize_decimal_text)
    normalized["spvxtstr"] = pd.to_numeric(numeric_text, errors="coerce")
    normalized = normalized.dropna(subset=["trade_date", "spvxtstr"])
    normalized = normalized.sort_values("trade_date").drop_duplicates(
        subset=["trade_date"], keep="last"
    )
    normalized["spvxtstr_return"] = normalized["spvxtstr"].pct_change()
    normalized["spvxtstr_index"] = normalized["spvxtstr"] / normalized["spvxtstr"].iloc[0]
    return normalized.set_index("trade_date")


def normalize_decimal_text(value: str) -> str:
    """Normalize common decimal/thousands separator variants into plain decimal text."""
    text = str(value).strip().replace(" ", "")
    if not text:
        return text

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")
        return text.replace(",", "")

    if "," in text:
        return text.replace(",", ".")

    return text


def _returns_frame(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        return returns.to_frame()
    return returns.copy()


def summarize_bucket_returns(
    returns: pd.Series | pd.DataFrame,
    dislocation_bucket: pd.Series,
    *,
    future_periods: int = 0,
    bucket_order: Sequence[str] = DEFAULT_BUCKET_ORDER,
) -> pd.DataFrame:
    """Summarize mean returns by dislocation bucket for same-day or future returns."""
    returns_frame = _returns_frame(returns).sort_index()
    if future_periods:
        returns_frame = returns_frame.shift(-future_periods)

    bucket_series = pd.Series(dislocation_bucket, copy=True).sort_index().rename(
        "dislocation_bucket"
    )
    aligned = returns_frame.join(bucket_series, how="inner")
    aligned = aligned.dropna(subset=["dislocation_bucket"]).dropna(how="any")

    value_columns = [column for column in aligned.columns if column != "dislocation_bucket"]
    grouped = aligned.groupby("dislocation_bucket")
    summary = grouped[value_columns].mean().reindex(bucket_order)
    counts = grouped.size().reindex(bucket_order)
    summary.insert(0, "count", counts)
    return summary


def summarize_conditional_returns(
    returns: pd.Series | pd.DataFrame, mask: pd.Series
) -> pd.DataFrame:
    """Summarize count, mean, and median returns on a selected subset of dates."""
    returns_frame = _returns_frame(returns).sort_index()
    condition = pd.Series(mask, copy=True).reindex(returns_frame.index).fillna(False)
    subset = returns_frame.loc[condition.astype(bool)].dropna(how="any")

    return pd.DataFrame(
        {
            "count": subset.count(),
            "mean": subset.mean(),
            "median": subset.median(),
        }
    )


def extract_extreme_moves(
    returns: pd.Series, mask: pd.Series, *, threshold: float = 0.03
) -> pd.DataFrame:
    """Return the subset of large positive or negative moves on selected dates."""
    series = pd.Series(returns, copy=True).sort_index().dropna()
    condition = pd.Series(mask, copy=True).reindex(series.index).fillna(False)
    subset = series.loc[condition.astype(bool)]
    subset = subset[subset.abs() >= threshold]

    extremes = subset.rename("return").to_frame()
    extremes["direction"] = np.where(extremes["return"] >= threshold, "positive", "negative")
    return extremes


def build_generic_return_frame(
    generic_frame: pd.DataFrame,
    *,
    return_field: str,
    start: str | pd.Timestamp | None = DEFAULT_START,
    end: str | pd.Timestamp | None = DEFAULT_END,
    symbols: Sequence[str] = DEFAULT_UX_SYMBOLS,
) -> pd.DataFrame:
    """Build a wide generic return frame from the cleaned VX ladder."""
    frame = generic_frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame[frame["ux_symbol"].isin(symbols)]

    returns = (
        frame.pivot(index="trade_date", columns="ux_symbol", values=return_field)
        .sort_index()
        .reindex(columns=list(symbols))
    )

    if start is not None or end is not None:
        returns = returns.loc[start:end]

    return returns


def build_return_frame_from_levels(level_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute returns directly from a wide level frame without forward filling gaps."""
    levels = level_frame.apply(pd.to_numeric, errors="coerce")
    returns = levels.pct_change(fill_method=None)
    valid_points = levels.notna() & levels.shift(1).notna()
    return returns.where(valid_points)


def build_spvxtstr_program_weights(signal_frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Build the two-asset S0-style ES/SPVXTSTR allocation weights."""
    es_weight = pd.to_numeric(signal_frame["es_weight_signal"], errors="coerce")
    spvxtstr_weight = 1.0 - es_weight

    weights = pd.DataFrame(index=signal_frame.index)
    weights["program"] = name
    weights["regime"] = "base"
    weights["weight_es"] = es_weight
    weights["weight_spvxtstr"] = spvxtstr_weight
    has_signal = es_weight.notna()
    weights["gross_exposure"] = np.where(
        has_signal,
        weights[["weight_es", "weight_spvxtstr"]].abs().sum(axis=1),
        np.nan,
    )
    weights["net_exposure"] = np.where(
        has_signal,
        weights[["weight_es", "weight_spvxtstr"]].sum(axis=1),
        np.nan,
    )
    return weights


def combine_weighted_returns(
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    asset_weight_map: dict[str, str],
    name: str,
) -> pd.Series:
    """Combine aligned asset returns using explicit weight-column mappings."""
    aligned_returns = asset_returns.reindex(weights.index)
    total_return = pd.Series(0.0, index=weights.index, dtype=float)
    valid_mask = pd.Series(True, index=weights.index)

    for asset_column, weight_column in asset_weight_map.items():
        asset_series = pd.to_numeric(aligned_returns[asset_column], errors="coerce")
        weight_series = pd.to_numeric(weights[weight_column], errors="coerce")
        total_return = total_return.add(weight_series * asset_series, fill_value=np.nan)
        valid_mask &= asset_series.notna() & weight_series.notna()

    total_return.loc[~valid_mask] = np.nan
    total_return.name = name
    return total_return


def build_growth_index(returns: pd.Series, *, start_value: float = 1.0) -> pd.Series:
    """Convert a return stream into a rebased growth index."""
    series = pd.to_numeric(pd.Series(returns, copy=True), errors="coerce")
    cumulative = (1.0 + series.fillna(0.0)).cumprod() * start_value
    cumulative.loc[series.isna()] = np.nan
    cumulative.name = getattr(returns, "name", None)
    return cumulative


def build_drawdown_series(growth_index: pd.Series) -> pd.Series:
    """Compute drawdowns from a rebased growth index."""
    index_series = pd.to_numeric(pd.Series(growth_index, copy=True), errors="coerce")
    running_peak = index_series.cummax()
    drawdown = index_series / running_peak - 1.0
    drawdown.loc[index_series.isna()] = np.nan
    drawdown.name = getattr(growth_index, "name", None)
    return drawdown


def build_turnover_series(
    weights: pd.DataFrame, *, weight_columns: Sequence[str]
) -> pd.Series:
    """Compute one-way turnover as half the absolute change in target weights."""
    weight_frame = weights.loc[:, list(weight_columns)].apply(pd.to_numeric, errors="coerce")
    turnover = 0.5 * weight_frame.diff().abs().sum(axis=1)
    valid_rows = weight_frame.notna().all(axis=1)
    prior_valid_rows = valid_rows.shift(1, fill_value=False)
    turnover.loc[~(valid_rows & prior_valid_rows)] = np.nan
    turnover.name = "turnover"
    return turnover


def summarize_performance(
    returns: pd.Series, *, turnover: pd.Series | None = None, annualization: int = 252
) -> pd.Series:
    """Summarize annualized return, risk, drawdown, and turnover for one return stream."""
    series = pd.to_numeric(pd.Series(returns, copy=True), errors="coerce").dropna()
    if series.empty:
        return pd.Series(
            {
                "annualized_return": np.nan,
                "annualized_volatility": np.nan,
                "max_drawdown": np.nan,
                "sharpe_ratio": np.nan,
                "sortino_ratio": np.nan,
                "turnover": np.nan,
                "observations": 0,
            }
        )

    periods = len(series)
    annualized_return = (1.0 + series).prod() ** (annualization / periods) - 1.0
    annualized_volatility = series.std(ddof=1) * np.sqrt(annualization)
    growth_index = build_growth_index(series)
    drawdown = build_drawdown_series(growth_index)
    max_drawdown = drawdown.min()
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility else np.nan

    downside_deviation = np.sqrt((series.clip(upper=0) ** 2).mean()) * np.sqrt(annualization)
    sortino_ratio = annualized_return / downside_deviation if downside_deviation else np.nan
    mean_turnover = pd.to_numeric(turnover, errors="coerce").mean() if turnover is not None else np.nan

    return pd.Series(
        {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "turnover": mean_turnover,
            "observations": periods,
        }
    )


def build_term_structure(
    generic_frame: pd.DataFrame,
    *,
    price_field: str,
    start: str | pd.Timestamp | None = DEFAULT_START,
    end: str | pd.Timestamp | None = DEFAULT_END,
    symbols: Sequence[str] = DEFAULT_UX_SYMBOLS,
    forward_fill: bool = False,
) -> pd.DataFrame:
    """Build a wide UX term structure from the generic futures ladder."""
    frame = generic_frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame[frame["ux_symbol"].isin(symbols)]

    curve = (
        frame.pivot(index="trade_date", columns="ux_symbol", values=price_field)
        .sort_index()
        .reindex(columns=list(symbols))
    )

    if forward_fill:
        curve = curve.ffill()

    if start is not None or end is not None:
        curve = curve.loc[start:end]

    return curve


def count_curve_dislocations(curve: pd.DataFrame) -> pd.Series:
    """Count term-structure dislocations using the paper's strict-lt ranking rule."""
    if curve.empty:
        return pd.Series(index=curve.index, dtype="Int64", name="dislocation_count")

    values = curve.to_numpy(dtype=float)
    dislocation_counts = np.full(len(curve), np.nan)
    valid_rows = ~np.isnan(values).any(axis=1)
    valid_values = values[valid_rows]

    if len(valid_values) > 0:
        lower_counts = (valid_values[:, None, :] < valid_values[:, :, None]).sum(axis=2)
        expected_ranks = lower_counts + 1
        canonical_ranks = np.arange(1, valid_values.shape[1] + 1)
        dislocation_counts[valid_rows] = (expected_ranks != canonical_ranks).sum(axis=1)

    return pd.Series(
        pd.array(dislocation_counts, dtype="Int64"),
        index=curve.index,
        name="dislocation_count",
    )


def bucket_dislocations(dislocation_count: pd.Series) -> pd.Series:
    """Map raw dislocation counts into the whitepaper's three buckets."""
    bucket = pd.Series(pd.NA, index=dislocation_count.index, dtype="object")
    bucket.loc[dislocation_count == 0] = PERFECT_CONTANGO_BUCKET
    bucket.loc[(dislocation_count >= 2) & (dislocation_count <= 4)] = (
        LOW_DISLOCATION_BUCKET
    )
    bucket.loc[dislocation_count >= 5] = HIGH_DISLOCATION_BUCKET
    bucket.name = "dislocation_bucket"
    return bucket


def es_weight_from_dislocations(
    dislocation_count: pd.Series, *, contract_count: int = 7
) -> pd.Series:
    """Convert dislocations into the paper's intended ES allocation weight."""
    weight = 1.0 - pd.to_numeric(dislocation_count, errors="coerce") / contract_count
    weight.name = "es_weight"
    return weight


def compute_curve_slope(curve: pd.DataFrame) -> pd.Series:
    """Compute the paper's UX1/UX7 slope measure."""
    slope = 1.0 - curve["UX1"] / curve["UX7"]
    slope.name = "slope"
    return slope


def lag_weekly_average(signal: pd.Series) -> pd.Series:
    """Map each trade date to the previous completed week's average signal."""
    if signal.empty:
        return signal.astype(float)

    weekly_periods = signal.index.to_period("W-FRI")
    weekly_average = signal.groupby(weekly_periods).mean().shift(1)
    lagged = pd.Series(weekly_periods, index=signal.index).map(weekly_average)
    lagged.name = signal.name
    return lagged.astype(float)


def build_daily_signal_frame(
    signal_curve: pd.DataFrame,
    *,
    vix_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the tradable daily signal set from the signal-generation UX curve."""
    dislocation_count = count_curve_dislocations(signal_curve)
    signal_frame = pd.DataFrame(index=signal_curve.index)
    signal_frame["dislocation_count_raw"] = dislocation_count
    signal_frame["dislocation_bucket_raw"] = bucket_dislocations(dislocation_count)
    signal_frame["es_weight_raw"] = es_weight_from_dislocations(dislocation_count)
    signal_frame["slope_raw"] = compute_curve_slope(signal_curve)
    signal_frame["es_weight_signal"] = signal_frame["es_weight_raw"].shift(1)
    signal_frame["slope_signal"] = signal_frame["slope_raw"].shift(1)

    if vix_close is not None:
        vix_history = pd.Series(vix_close, dtype=float).copy()
        vix_history.index = pd.to_datetime(vix_history.index)
        signal_frame["vix_close_raw"] = vix_history.reindex(signal_frame.index)
        signal_frame["vix_close_signal"] = signal_frame["vix_close_raw"].shift(1)

    return signal_frame


def build_weekly_signal_frame(
    signal_curve: pd.DataFrame,
    *,
    vix_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the weekly signal set using previous-week averages."""
    dislocation_count = count_curve_dislocations(signal_curve)
    slope_raw = compute_curve_slope(signal_curve)

    signal_frame = pd.DataFrame(index=signal_curve.index)
    signal_frame["dislocation_count_raw"] = dislocation_count
    signal_frame["dislocation_bucket_raw"] = bucket_dislocations(dislocation_count)
    signal_frame["es_weight_raw"] = es_weight_from_dislocations(dislocation_count)
    signal_frame["slope_raw"] = slope_raw
    signal_frame["weekly_dislocation_mean_signal"] = lag_weekly_average(
        dislocation_count.astype(float)
    )
    signal_frame["es_weight_signal"] = es_weight_from_dislocations(
        signal_frame["weekly_dislocation_mean_signal"]
    )
    signal_frame["weekly_slope_mean_signal"] = lag_weekly_average(slope_raw)
    signal_frame["slope_signal"] = signal_frame["weekly_slope_mean_signal"]

    if vix_close is not None:
        vix_history = pd.Series(vix_close, dtype=float).copy()
        vix_history.index = pd.to_datetime(vix_history.index)
        signal_frame["vix_close_raw"] = vix_history.reindex(signal_frame.index)
        signal_frame["weekly_vix_close_mean_signal"] = lag_weekly_average(
            signal_frame["vix_close_raw"]
        )
        signal_frame["vix_close_signal"] = signal_frame[
            "weekly_vix_close_mean_signal"
        ]

    return signal_frame


def build_base_program_weights(signal_frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Convert ES signal weights into investable ES, UX1, and UX3 exposures."""
    es_weight = pd.to_numeric(signal_frame["es_weight_signal"], errors="coerce")
    volatility_weight = 1.0 - es_weight

    weights = pd.DataFrame(index=signal_frame.index)
    weights["program"] = name
    weights["regime"] = "base"
    weights["weight_es"] = es_weight
    weights["weight_va"] = volatility_weight
    weights["weight_ux1"] = -0.5 * volatility_weight
    weights["weight_ux3"] = volatility_weight

    has_signal = es_weight.notna()
    weights["gross_exposure"] = np.where(
        has_signal,
        weights[["weight_es", "weight_ux1", "weight_ux3"]].abs().sum(axis=1),
        np.nan,
    )
    weights["net_exposure"] = np.where(
        has_signal,
        weights[["weight_es", "weight_ux1", "weight_ux3"]].sum(axis=1),
        np.nan,
    )
    return weights


def build_refined_program_weights(
    signal_frame: pd.DataFrame,
    *,
    name: str,
    slope_threshold: float,
    vix_threshold: float,
    allow_short_es: bool,
) -> pd.DataFrame:
    """Apply the paper's slope and spot-VIX refinements to the base program."""
    weights = build_base_program_weights(signal_frame, name=name)
    es_weight = weights["weight_es"].copy()
    volatility_weight = weights["weight_va"].copy()
    slope_signal = pd.to_numeric(signal_frame["slope_signal"], errors="coerce")

    if "vix_close_signal" in signal_frame:
        vix_close_signal = pd.to_numeric(signal_frame["vix_close_signal"], errors="coerce")
    else:
        vix_close_signal = pd.Series(np.nan, index=signal_frame.index)

    dampened = es_weight.notna() & slope_signal.lt(0.0) & slope_signal.gt(slope_threshold)
    es_weight.loc[dampened] = 0.5 * es_weight.loc[dampened] + 0.5
    volatility_weight.loc[dampened] = 0.5 * volatility_weight.loc[dampened]
    weights.loc[dampened, "regime"] = "dampened"

    doubled = (
        es_weight.notna()
        & slope_signal.gt(0.0)
        & vix_close_signal.gt(vix_threshold)
    )
    if allow_short_es:
        es_weight.loc[doubled] = 2.0 * weights.loc[doubled, "weight_es"] - 1.0
        volatility_weight.loc[doubled] = 2.0 * weights.loc[doubled, "weight_va"]
    else:
        doubled_es_weight = (2.0 * weights.loc[doubled, "weight_es"] - 1.0).clip(
            lower=0.0
        )
        es_weight.loc[doubled] = doubled_es_weight
        volatility_weight.loc[doubled] = 1.0 - doubled_es_weight
    weights.loc[doubled, "regime"] = "double_vol"

    weights["weight_es"] = es_weight
    weights["weight_va"] = volatility_weight
    weights["weight_ux1"] = -0.5 * volatility_weight
    weights["weight_ux3"] = volatility_weight
    has_signal = es_weight.notna()
    weights["gross_exposure"] = np.where(
        has_signal,
        weights[["weight_es", "weight_ux1", "weight_ux3"]].abs().sum(axis=1),
        np.nan,
    )
    weights["net_exposure"] = np.where(
        has_signal,
        weights[["weight_es", "weight_ux1", "weight_ux3"]].sum(axis=1),
        np.nan,
    )
    return weights


__all__ = [
    "CBOE_VIX_HISTORY_URL",
    "DEFAULT_BUCKET_ORDER",
    "DEFAULT_END",
    "DEFAULT_START",
    "DEFAULT_UX_SYMBOLS",
    "HIGH_DISLOCATION_BUCKET",
    "LOW_DISLOCATION_BUCKET",
    "PERFECT_CONTANGO_BUCKET",
    "build_base_program_weights",
    "build_daily_signal_frame",
    "build_drawdown_series",
    "build_generic_return_frame",
    "build_growth_index",
    "build_refined_program_weights",
    "build_return_frame_from_levels",
    "build_spvxtstr_program_weights",
    "build_term_structure",
    "build_turnover_series",
    "build_weekly_signal_frame",
    "bucket_dislocations",
    "combine_weighted_returns",
    "compute_curve_slope",
    "count_curve_dislocations",
    "es_weight_from_dislocations",
    "extract_extreme_moves",
    "lag_weekly_average",
    "load_spvxtstr_history",
    "load_spot_vix_history",
    "normalize_decimal_text",
    "summarize_performance",
    "summarize_bucket_returns",
    "summarize_conditional_returns",
]
