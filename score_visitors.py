#!/usr/bin/env python3
"""
Score (location, week) by predicted visitors — fewer visitors = higher score.

INPUT CSV (long format)
-----------------------
Must contain columns:
- location OR station   (string/int)
- week                  (int: 1..15 or any week index you use)
- visitors              (float/int: predicted visitors)

Example:
location,week,visitors
MtHotham,7,1340
FallsCreek,7,2110
...

Scoring
-------
We standardize visitor counts across the whole file and invert them:
    visitor_score = -z(visitors)
So lower visitors => larger (better) score.
If all visitor values are identical, scores become 0 (tie).

Output
------
A ranked CSV with columns:
rank, station, week, visitors, visitor_score
"""

from __future__ import annotations
import argparse
from typing import Optional, List
import numpy as np
import pandas as pd


# ---------- utils ----------
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column; expected one of: {candidates}")
    return ""


def load_visitors_csv(path: str,
                      station_col: Optional[str] = None,
                      week_col: Optional[str] = None,
                      visitors_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Flexible column detection
    station_col = station_col or _pick_col(
        df, ["station", "location", "site", "resort", "mountain"]
    )
    week_col = week_col or _pick_col(df, ["week", "trip_week", "wk"])
    visitors_col = visitors_col or _pick_col(
        df, ["visitors", "predicted_visitors", "visitor_count", "count", "n_visitors"]
    )

    # Normalize column names
    df = df.rename(columns={station_col: "station", week_col: "week", visitors_col: "visitors"})

    # Types & cleaning
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["visitors"] = pd.to_numeric(df["visitors"], errors="coerce")
    df = df.dropna(subset=["week", "visitors"]).copy()
    df["week"] = df["week"].astype(int)

    # If there are duplicate (station, week) rows, average them
    df = (
        df.groupby(["station", "week"], as_index=False)["visitors"]
          .mean()
          .sort_values(["station", "week"])
          .reset_index(drop=True)
    )
    return df


def score_visitors(df: pd.DataFrame) -> pd.DataFrame:
    z = _zscore(df["visitors"].values)
    df = df.copy()
    df["visitor_score"] = -z  # invert: fewer visitors => higher score
    # Rank: higher score first; break ties by lower absolute visitors
    df = df.sort_values(by=["visitor_score", "visitors"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df[["rank", "station", "week", "visitors", "visitor_score"]]


def main():
    ap = argparse.ArgumentParser(description="Score and rank (location, week) by predicted visitors — fewer is better.")
    ap.add_argument("--visitors", required=True, help="Path to visitor CSV (long format).")
    ap.add_argument("--out", default="visitor_scores.csv", help="Output CSV path.")
    ap.add_argument("--top", type=int, default=20, help="How many rows to print to console.")
    # Optional explicit column names if your headers are unusual:
    ap.add_argument("--station-col", default=None, help="Name of the station/location column.")
    ap.add_argument("--week-col", default=None, help="Name of the week column.")
    ap.add_argument("--visitors-col", default=None, help="Name of the visitors column.")
    args = ap.parse_args()

    df = load_visitors_csv(
        args.visitors,
        station_col=args.station_col,
        week_col=args.week_col,
        visitors_col=args.visitors_col,
    )
    scored = score_visitors(df)

    # Print a preview
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(scored.head(args.top).to_string(index=False))

    # Save full table
    scored.to_csv(args.out, index=False)
    print(f"\nSaved full visitor ranking to: {args.out}")

    # One-line winner
    best = scored.iloc[0]
    print(
        f"Quietest pick → station {best.station}, week {int(best.week)} "
        f"| visitors {best.visitors:.0f}, visitor_score {best.visitor_score:.3f}"
    )


if __name__ == "__main__":
    main()
