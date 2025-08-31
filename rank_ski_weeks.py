#!/usr/bin/env python3
"""
Rank ski-trip options using:
  - Heavy snow the week before (bigger = better)
  - Lighter snow during the trip (smaller = better)
  - DROP any (station, week) whose *trip-week* minimum temperature > threshold (default -1°C)

Inputs
------
1) --csv   : predictions CSV with columns: station, 1..15
2) --temps : min-temp   CSV with columns: station, 1..15  (°C)

Scoring
-------
score = w_prev*z(prev_week) + w_drop*z(prev - curr) + w_curr*(-z(curr_week))

Usage
-----
python rank_ski_weeks.py --csv predictions.csv --temps min_temps.csv --top 20
# Only consider week 9 trip:
python rank_ski_weeks.py --csv predictions.csv --temps min_temps.csv --target-week 9
# Adjust weights or the temperature threshold:
python rank_ski_weeks.py --csv predictions.csv --temps min_temps.csv --w-prev 0.6 --w-drop 0.2 --w-curr 0.2 --temp-threshold -1
"""

from __future__ import annotations
import argparse
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


def _zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def _coerce_week_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Detect and order week columns as ints 1..N."""
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


def load_matrix(path: str) -> Tuple[pd.DataFrame, List[int]]:
    df = pd.read_csv(path)
    if "station" not in df.columns:
        raise ValueError(f"{path} must include a 'station' column.")
    df, weeks = _coerce_week_cols(df)
    return df, weeks


def build_candidate_table(
    snow_df: pd.DataFrame,
    weeks: List[int],
    temp_df: pd.DataFrame,
    temp_threshold: float,
) -> pd.DataFrame:
    """
    Build one row per (station, trip_week) with prev/curr snow, drop, and trip-week min temp.
    Rows with trip-week min temp > temp_threshold are filtered OUT.
    """
    temp_idx = temp_df.set_index("station")
    records = []

    for w in weeks:
        if w == weeks[0]:
            # No previous week for the very first week
            continue
        prev_col, curr_col = str(w - 1), str(w)
        for _, row in snow_df.iterrows():
            station = row["station"]
            prev_val = row[prev_col]
            curr_val = row[curr_col]

            # look up trip-week min temp
            try:
                trip_min_temp = float(temp_idx.loc[station, curr_col])
            except Exception:
                trip_min_temp = np.nan

            # Filter: require a valid, cold-enough trip week
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
    """Add score components and overall score; larger = better."""
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
    cand["score_curr"] = weights[2] * (-z_curr)  # lower current snow preferred
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
    return scored[cols].head(top_k), cand, scored  # return also full tables


def main():
    p = argparse.ArgumentParser(description="Rank ski trip weeks by snow pattern with temperature filtering.")
    p.add_argument("--csv", required=True, help="Path to predictions CSV (station, 1..15).")
    p.add_argument("--temps", required=True, help="Path to min-temp CSV (station, 1..15).")
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

    snow_df, weeks = load_matrix(args.csv)
    temp_df, temp_weeks = load_matrix(args.temps)

    # Basic sanity: overlapping week sets
    if set(weeks) - set(temp_weeks):
        missing = sorted(set(weeks) - set(temp_weeks))
        raise ValueError(f"Temperature file missing week columns: {missing}")

    top, cand_filtered, scored_full = recommend(
        snow_df,
        weeks,
        temp_df=temp_df,
        temp_threshold=args.temp_threshold,
        target_week=args.target_week,
        w_prev=args.w_prev,
        w_drop=args.w_drop,
        w_curr=args.w_curr,
        top_k=args.top,
    )

    # Print the shortlist
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(top.to_string(index=False))

    # Save the full ranking (post-filter)
    scored_full.to_csv(args.out, index=False)
    print(f"\nSaved full ranking (after temp filter) to: {args.out}")

    # Friendly one-line winner summary
    best = scored_full.iloc[0]
    print(
        f"Best overall → station {best.station}, week {int(best.trip_week)} "
        f"(prev {int(best.prev_week)}), score {best.score:.3f} | "
        f"prev {best.prev_snow:.2f}, curr {best.curr_snow:.2f}, drop {best.drop_snow:.2f}, "
        f"trip min temp {best.trip_min_temp:.1f}°C"
    )


if __name__ == "__main__":
    main()
