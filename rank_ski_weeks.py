#!/usr/bin/env python3
"""
Rank ski-trip options: choose (station, week) where the *previous* week had heavy
snow and the *trip* week has lighter snow.

Inputs
------
CSV with columns:
- 'station' (string or int id)
- week numbers 1..15 (floats: avg daily rainfall/snow)

Scoring (defaults can be changed with CLI flags)
-------
score = w_prev*z(prev_week) + w_drop*z(prev - curr) + w_curr*(-z(curr_week))

Where z() is the z-score across all stations and weeks (so every pair competes
against *all* others). Ties favor lower current-week snow, then higher prev-week snow.

Usage
-----
python rank_ski_weeks.py --csv predictions.csv --top 20
# Only consider a specific trip week (e.g., week 9):
python rank_ski_weeks.py --csv predictions.csv --target-week 9
# Adjust weights:
python rank_ski_weeks.py --csv predictions.csv --w-prev 0.6 --w-drop 0.2 --w-curr 0.2
"""

from __future__ import annotations
import argparse
import math
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Safe z-score with NaN handling and zero-variance guard."""
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def _coerce_week_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Make week columns ints 1..N in order."""
    # Map columns that look like ints (e.g., "1", 1) into ints
    week_cols = []
    for c in df.columns:
        if c == "station":
            continue
        try:
            week = int(str(c).strip())
            week_cols.append(week)
        except Exception:
            pass
    if not week_cols:
        raise ValueError("No week columns detected. Expect columns named 1..15.")
    week_cols = sorted(week_cols)
    # Rebuild df with ordered week columns
    keep = ["station"] + [str(w) if str(w) in df.columns else w for w in week_cols]
    df = df[keep].copy()
    # Ensure numeric
    for w in week_cols:
        col = str(w) if str(w) in df.columns else w
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, week_cols


def load_predictions(path: str) -> Tuple[pd.DataFrame, List[int]]:
    df = pd.read_csv(path)
    if "station" not in df.columns:
        raise ValueError("CSV must include a 'station' column.")
    df, weeks = _coerce_week_cols(df)
    return df, weeks


def build_candidate_table(df: pd.DataFrame, weeks: List[int]) -> pd.DataFrame:
    """
    Return long-form table with one row per (station, trip_week):
      - prev_week_value, curr_week_value, diff (prev - curr)
    Only weeks with a valid previous week are included (i.e., weeks[0] is skipped).
    """
    records = []
    for w in weeks:
        if w == weeks[0]:
            # No previous week available for the very first week
            continue
        prev_col, curr_col = str(w - 1), str(w)
        for _, row in df.iterrows():
            station = row["station"]
            prev_val = row[prev_col]
            curr_val = row[curr_col]
            if pd.isna(prev_val) or pd.isna(curr_val):
                continue
            records.append(
                {
                    "station": station,
                    "trip_week": int(w),
                    "prev_week": int(w - 1),
                    "prev_snow": float(prev_val),
                    "curr_snow": float(curr_val),
                    "drop_snow": float(prev_val - curr_val),
                }
            )
    cand = pd.DataFrame.from_records(records)
    if cand.empty:
        raise ValueError("No valid (station, week) pairs found after preprocessing.")
    return cand


def score_candidates(
    cand: pd.DataFrame,
    w_prev: float = 0.6,
    w_drop: float = 0.2,
    w_curr: float = 0.2,
) -> pd.DataFrame:
    """
    Add a 'score' column using standardized components across *all* candidates.
    Larger = better:
      - Higher previous-week snow (fresh base)
      - Larger drop from prev->current (fresh snow but clearing)
      - Lower current-week snow (more sun) via negative z of curr_snow
    """
    # Normalize weights to sum to 1
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
    cand["score_curr"] = weights[2] * (-z_curr)  # lower current snow is better
    cand["score"] = cand["score_prev"] + cand["score_drop"] + cand["score_curr"]

    # Helpful tie-breakers: prefer lower current week, then higher previous week
    cand = cand.sort_values(
        by=["score", "curr_snow", "prev_snow"], ascending=[False, True, False]
    ).reset_index(drop=True)
    return cand


def recommend(
    df: pd.DataFrame,
    weeks: List[int],
    target_week: Optional[int] = None,
    w_prev: float = 0.6,
    w_drop: float = 0.2,
    w_curr: float = 0.2,
    top_k: int = 20,
) -> pd.DataFrame:
    cand = build_candidate_table(df, weeks)
    if target_week is not None:
        if int(target_week) not in cand["trip_week"].unique():
            raise ValueError(f"target_week {target_week} not present in data.")
        cand = cand[cand["trip_week"] == int(target_week)].copy()

    scored = score_candidates(cand, w_prev=w_prev, w_drop=w_drop, w_curr=w_curr)

    cols = [
        "rank",
        "station",
        "trip_week",
        "prev_week",
        "prev_snow",
        "curr_snow",
        "drop_snow",
        "score",
        "score_prev",
        "score_drop",
        "score_curr",
    ]
    scored = scored.reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored[cols].head(top_k)


def main():
    p = argparse.ArgumentParser(description="Rank ski trip weeks by snow pattern.")
    p.add_argument("--csv", required=True, help="Path to predictions CSV.")
    p.add_argument("--target-week", type=int, default=None,
                   help="If set, only evaluate this trip week (uses week-1 as 'prev').")
    p.add_argument("--top", type=int, default=20, help="How many results to print/save.")
    p.add_argument("--w-prev", type=float, default=0.6, help="Weight for previous-week snow.")
    p.add_argument("--w-drop", type=float, default=0.2, help="Weight for (prev - curr).")
    p.add_argument("--w-curr", type=float, default=0.2, help="Weight for lower current-week snow.")
    p.add_argument("--out", default="ski_week_recommendations.csv",
                   help="Optional CSV to write results to.")
    args = p.parse_args()

    df, weeks = load_predictions(args.csv)
    recs = recommend(
        df,
        weeks,
        target_week=args.target_week,
        w_prev=args.w_prev,
        w_drop=args.w_drop,
        w_curr=args.w_curr,
        top_k=args.top,
    )

    # Print nicely
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(recs.to_string(index=False))

    # Save full ranking (not just top_k)
    cand_full = build_candidate_table(df, weeks)
    scored_full = score_candidates(
        cand_full, w_prev=args.w_prev, w_drop=args.w_drop, w_curr=args.w_curr
    )
    scored_full.to_csv(args.out, index=False)
    print(f"\nSaved full ranking to: {args.out}")


if __name__ == "__main__":
    main()
