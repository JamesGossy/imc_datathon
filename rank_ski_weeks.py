#!/usr/bin/env python3
"""
Rank ski-trip options using:
  - Heavy snow the week before (bigger = better)
  - Lighter snow during the trip (smaller = better)
  - Filter out any (station, trip_week) whose trip-week MIN temperature > threshold (default -1°C)
  - Temperature CSV may include BOTH Tmin and Tmax; we auto-detect and use ONLY Tmin.

Inputs
------
--csv   : rainfall/snow predictions (wide): station, 1..15
--temps : Tmin/Tmax (wide): station, week-level columns that include *min* or *tmin* in their names
         Examples handled: 1_min, min_1, tmin_1, week1_min, wk_01_tmin, etc.

Scoring
-------
score = w_prev*z(prev_week) + w_drop*z(prev - curr) + w_curr*(-z(curr_week))
"""

from __future__ import annotations
import argparse
import re
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd


# ---------- helpers ----------
def _zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def _coerce_week_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Detect and order week columns as ints 1..N for a matrix shaped like: station, 1..15."""
    week_cols = []
    for c in df.columns:
        if c == "station":
            continue
        try:
            week_cols.append(int(str(c).strip()))
        except Exception:
            pass
    if not week_cols:
        raise ValueError("No week columns detected. Expect columns named 1..15.")
    week_cols = sorted(week_cols)
    keep = ["station"] + [str(w) if str(w) in df.columns else w for w in week_cols]
    df = df[keep].copy()
    for w in week_cols:
        col = str(w) if str(w) in df.columns else w
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, week_cols


def load_snow_matrix(path: str) -> Tuple[pd.DataFrame, List[int]]:
    df = pd.read_csv(path)
    if "station" not in df.columns:
        raise ValueError(f"{path} must include a 'station' column.")
    return _coerce_week_cols(df)


def _extract_tmin_columns(temp_df: pd.DataFrame) -> Dict[int, str]:
    """
    From a wide temp dataframe with both Tmin and Tmax columns, pick ONLY Tmin.
    We look for columns whose names contain 'tmin' or a token 'min' (case-insensitive),
    and explicitly exclude anything containing 'tmax' or 'max'.
    We also parse the week number from the column name.
    """
    week_to_col: Dict[int, str] = {}

    for col in temp_df.columns:
        if col == "station":
            continue
        name = str(col)
        low = name.lower()

        # must include tmin or a clean 'min' token; exclude 'tmax'/'max'
        is_min = ("tmin" in low) or bool(re.search(r"(?:^|[_\-\s])min(?:$|[_\-\s])", low))
        is_max = ("tmax" in low) or bool(re.search(r"(?:^|[_\-\s])max(?:$|[_\-\s])", low))
        if not is_min or is_max:
            continue

        m = re.search(r"(\d+)", low)   # first integer in the name = week number
        if not m:
            continue
        wk = int(m.group(1))

        # Prefer a column explicitly containing 'tmin' over generic 'min' if duplicates
        if wk not in week_to_col:
            week_to_col[wk] = col
        else:
            if "tmin" in low and "tmin" not in str(week_to_col[wk]).lower():
                week_to_col[wk] = col

    return week_to_col


def load_tmin_matrix(path: str) -> Tuple[pd.DataFrame, List[int]]:
    """
    Read a wide Tmin/Tmax file and return a dataframe shaped like: station, 1..15 (Tmin only).
    """
    df = pd.read_csv(path)
    if "station" not in df.columns:
        raise ValueError(f"{path} must include a 'station' column.")

    week_to_col = _extract_tmin_columns(df)

    if not week_to_col:
        # Fallback: maybe the file already has only Tmin with plain week numbers 1..15
        try:
            df2, weeks = _coerce_week_cols(df.copy())
            return df2, weeks
        except Exception as e:
            raise ValueError(
                "Could not detect Tmin columns. Make sure Tmin columns include 'tmin' "
                "or a clean 'min' token (e.g., '1_min', 'tmin_1', 'week1_min')."
            ) from e

    weeks = sorted(week_to_col.keys())
    tmin_cols = ["station"] + [week_to_col[w] for w in weeks]
    out = df[tmin_cols].copy()

    # Rename columns to plain week numbers 1..N and ensure numeric
    rename_map = {week_to_col[w]: str(w) for w in weeks}
    out = out.rename(columns=rename_map)
    for w in weeks:
        out[str(w)] = pd.to_numeric(out[str(w)], errors="coerce")

    return out, weeks


# ---------- pipeline ----------
def build_candidate_table(
    snow_df: pd.DataFrame,
    weeks: List[int],
    temp_df: pd.DataFrame,
    temp_threshold: float,
) -> pd.DataFrame:
    """Build candidate rows and apply Tmin filter."""
    temp_idx = temp_df.set_index("station")
    records = []

    for w in weeks:
        if w == weeks[0]:
            continue  # no previous week for the first week
        prev_col, curr_col = str(w - 1), str(w)
        for _, row in snow_df.iterrows():
            station = row["station"]
            prev_val = row[prev_col]
            curr_val = row[curr_col]

            try:
                trip_min_temp = float(temp_idx.loc[station, curr_col])
            except Exception:
                trip_min_temp = np.nan

            if pd.isna(prev_val) or pd.isna(curr_val) or pd.isna(trip_min_temp):
                continue
            if trip_min_temp > temp_threshold:
                continue

            records.append(
                {
                    "station": station,
                    "trip_week": int(w),
                    "prev_week": int(w - 1),
                    "prev_snow": float(prev_val),
                    "curr_snow": float(curr_val),
                    "drop_snow": float(prev_val - curr_val),
                    "trip_min_temp": trip_min_temp,
                }
            )

    cand = pd.DataFrame.from_records(records)
    if cand.empty:
        raise ValueError(
            "No valid (station, week) pairs after temperature filtering. "
            "Try relaxing --temp-threshold or check your inputs."
        )
    return cand


def score_candidates(
    cand: pd.DataFrame,
    w_prev: float = 0.6,
    w_drop: float = 0.2,
    w_curr: float = 0.2,
) -> pd.DataFrame:
    weights = np.array([w_prev, w_drop, w_curr], dtype=float)
    if np.any(weights < 0) or np.all(weights == 0):
        raise ValueError("Weights must be non-negative and not all zero.")
    weights = weights / weights.sum()

    z_prev = _zscore(cand["prev_snow"].values)
    z_drop = _zscore(cand["drop_snow"].values)
    z_curr = _zscore(cand["curr_snow"].values)

    cand = cand.copy()
    cand["score_prev"] = weights[0] * z_prev
    cand["score_drop"] = weights[1] * z_drop
    cand["score_curr"] = weights[2] * (-z_curr)  # prefer lower current snow
    cand["score"] = cand["score_prev"] + cand["score_drop"] + cand["score_curr"]

    cand = cand.sort_values(
        by=["score", "curr_snow", "prev_snow"], ascending=[False, True, False]
    ).reset_index(drop=True)
    return cand


def recommend(
    snow_df: pd.DataFrame,
    weeks: List[int],
    temp_df: pd.DataFrame,
    temp_threshold: float = -1.0,
    target_week: Optional[int] = None,
    w_prev: float = 0.6,
    w_drop: float = 0.2,
    w_curr: float = 0.2,
    top_k: int = 20,
) -> pd.DataFrame:
    cand = build_candidate_table(
        snow_df, weeks, temp_df=temp_df, temp_threshold=temp_threshold
    )
    if target_week is not None:
        target_week = int(target_week)
        if target_week not in cand["trip_week"].unique():
            raise ValueError(
                f"target_week {target_week} not present after filtering. "
                "Try a different week or relax --temp-threshold."
            )
        cand = cand[cand["trip_week"] == target_week].copy()

    scored = score_candidates(cand, w_prev=w_prev, w_drop=w_drop, w_curr=w_curr)

    cols = [
        "rank",
        "station",
        "trip_week",
        "prev_week",
        "prev_snow",
        "curr_snow",
        "drop_snow",
        "trip_min_temp",
        "score",
        "score_prev",
        "score_drop",
        "score_curr",
    ]
    scored = scored.reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored[cols].head(top_k), cand, scored


def main():
    p = argparse.ArgumentParser(description="Rank ski trip weeks by snow pattern with Tmin filtering; ignore Tmax.")
    p.add_argument("--csv", required=True, help="Path to predictions CSV (station, 1..15).")
    p.add_argument("--temps", required=True, help="Path to Tmin/Tmax CSV; ONLY Tmin will be used.")
    p.add_argument("--target-week", type=int, default=None,
                   help="If set, only evaluate this trip week (uses week-1 as 'prev').")
    p.add_argument("--top", type=int, default=20, help="How many results to print/save.")
    p.add_argument("--w-prev", type=float, default=0.6, help="Weight for previous-week snow.")
    p.add_argument("--w-drop", type=float, default=0.2, help="Weight for (prev - curr).")
    p.add_argument("--w-curr", type=float, default=0.2, help="Weight for lower current-week snow.")
    p.add_argument("--temp-threshold", type=float, default=-1.0,
                   help="Filter out trip weeks with min temp > this (°C). Default: -1.")
    p.add_argument("--out", default="ski_week_recommendations.csv",
                   help="CSV to write full ranked results.")
    args = p.parse_args()

    snow_df, weeks = load_snow_matrix(args.csv)
    tmin_df, tmin_weeks = load_tmin_matrix(args.temps)

    # Basic sanity: overlapping week sets
    missing = sorted(set(weeks) - set(tmin_weeks))
    if missing:
        raise ValueError(f"Temperature file missing Tmin for weeks: {missing}")

    top, cand_filtered, scored_full = recommend(
        snow_df,
        weeks,
        temp_df=tmin_df,
        temp_threshold=args.temp_threshold,
        target_week=args.target_week,
        w_prev=args.w_prev,
        w_drop=args.w_drop,
        w_curr=args.w_curr,
        top_k=args.top,
    )

    # Print shortlist
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(top.to_string(index=False))

    # Save full ranking (post-filter)
    scored_full.to_csv(args.out, index=False)
    print(f"\nSaved full ranking (after Tmin filter) to: {args.out}")

    # One-line winner summary
    best = scored_full.iloc[0]
    print(
        f"Best overall → station {best.station}, week {int(best.trip_week)} "
        f"(prev {int(best.prev_week)}), score {best.score:.3f} | "
        f"prev {best.prev_snow:.2f}, curr {best.curr_snow:.2f}, drop {best.drop_snow:.2f}, "
        f"trip Tmin {best.trip_min_temp:.1f}°C"
    )


if __name__ == "__main__":
    main()
