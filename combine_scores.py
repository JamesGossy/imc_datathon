#!/usr/bin/env python3
"""
Combine snow/visitor/rating/cost into a single weighted score and output:
- Neutral (overall) pick
- Budget family (CHEAP by price quantile)
- Wealthy family (EXPENSIVE by price quantile)

Weights (default): rain=0.5, visitors=0.3, rating=0.1, cost=0.1 (renormalized if changed)

Handles:
- Rain file with numeric station IDs (e.g., 71075.0) via --station-map
- Visitors long file (station, week, visitor_score) OR raw visitors (auto-inverted)
- Ratings (e.g., 'Mountain,Value' 1..5)
- Prices long OR one-row wide (resorts as columns)

De-duplicates:
- If multiple rows exist for the same (station, week), collapse using --rain-agg / --vis-agg (mean|max|median).
"""

from __future__ import annotations
import argparse
import re
from typing import List, Dict
import numpy as np
import pandas as pd


# ---------------- name normalization ----------------
def _norm_station(s: str) -> str:
    """Normalize resort/station names so 'Perisher AWS' == 'Perisher', 'Mt.' == 'Mount', etc."""
    if pd.isna(s):
        return ""
    s0 = str(s).strip().lower()

    # standardize 'mt.' / 'mt' -> 'mount'
    s0 = re.sub(r"\bmt\.?\b", "mount", s0)

    # remove BOM/telemetry tokens so weather-station names match resort names
    STOP_TOKENS = {"aws", "smhea", "bom", "automatic", "weather", "station", "met", "obs"}
    s0 = re.sub(r"[^\w\s]", " ", s0)      # keep letters/numbers/spaces
    s0 = re.sub(r"\s+", " ", s0).strip()
    s0 = " ".join(t for t in s0.split() if t not in STOP_TOKENS)

    # known typos/variants
    fixes = {
        "charlotte pas": "charlotte pass",
        "charlotte-pass": "charlotte pass",
        "charlottepass": "charlotte pass",
    }
    s0 = fixes.get(s0, s0)
    return s0


def _add_key(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out["station_key"] = out[col].astype(str).map(_norm_station)
    return out


# ---------------- utilities ----------------
def _pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Missing required column; expected one of: {candidates}. Found: {list(df.columns)}")
    return ""


def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.full(len(s), 0.5), index=s.index, dtype=float)
    vmin, vmax = float(s.min()), float(s.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return pd.Series(np.full(len(s), 0.5), index=s.index, dtype=float)
    return (s - vmin) / (vmax - vmin)


def _is_mostly_numeric(vals) -> bool:
    s = pd.Series(vals).astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return pd.to_numeric(s, errors="coerce").notna().mean() >= 0.9


def _agg_name_to_func(name: str):
    name = (name or "mean").lower()
    if name in ("mean", "avg"): return "mean"
    if name in ("median",): return "median"
    if name in ("max",): return "max"
    raise ValueError(f"Unsupported aggregator '{name}'. Use mean|median|max.")


def _collapse_duplicates(df: pd.DataFrame, keys: List[str], numeric_aggs: Dict[str, str], prefer_first_cols: List[str]):
    """
    Group by `keys` and aggregate:
      - numeric_aggs for numeric columns by name
      - 'first' for prefer_first_cols (e.g., 'station')
      - for other leftover numeric cols, default to 'mean'
    """
    df = df.copy()
    agg = {}
    for c in df.columns:
        if c in keys:
            continue
        if c in numeric_aggs:
            agg[c] = numeric_aggs[c]
        elif c in prefer_first_cols:
            agg[c] = "first"
        else:
            agg[c] = "mean" if pd.api.types.is_numeric_dtype(df[c]) else "first"
    return df.groupby(keys, as_index=False).agg(agg)


# ---------------- loaders ----------------
def _load_rain(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    station_col = _pick_col(df, ["station", "location", "resort", "mountain", "site"])
    week_col    = _pick_col(df, ["trip_week", "week", "wk"])
    score_col   = _pick_col(df, ["score", "rain_score", "overall_score", "snow_score"])
    out = df[[station_col, week_col, score_col]].copy()
    out.columns = ["station", "week", "rain_score"]
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64").astype(int)
    out["rain_score"] = pd.to_numeric(out["rain_score"], errors="coerce")
    out = out.dropna(subset=["station", "week", "rain_score"]).copy()
    return out


def _load_visitors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    station_col = _pick_col(df, ["station", "location", "resort", "mountain", "site"])
    week_col    = _pick_col(df, ["week", "trip_week", "wk"])
    vscore_col  = _pick_col(df, ["visitor_score", "quietness_score", "score"], required=False)

    if vscore_col:
        out = df[[station_col, week_col, vscore_col]].copy()
        out.columns = ["station", "week", "visitor_score"]
    else:
        visitors_col = _pick_col(df, ["visitors", "predicted_visitors", "visitor_count", "count", "n_visitors"])
        out = df[[station_col, week_col, visitors_col]].copy()
        out.columns = ["station", "week", "visitors"]
        out["visitors"] = pd.to_numeric(out["visitors"], errors="coerce")
        out["visitor_score"] = 1 - _minmax(out["visitors"])

    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64").astype(int)
    out["visitor_score"] = pd.to_numeric(out["visitor_score"], errors="coerce")
    out = out.dropna(subset=["station", "week", "visitor_score"]).copy()
    return out


def _load_ratings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    station_col = _pick_col(df, ["station", "location", "resort", "mountain", "site", "mountain"])
    rating_col  = _pick_col(df, ["rating", "tripadvisor", "trip_advisor", "tripadvisor_rating", "stars", "value"])
    week_present = any(c.lower() == "week" for c in df.columns)

    if week_present:
        week_col = _pick_col(df, ["week"])
        out = df[[station_col, week_col, rating_col]].copy()
        out.columns = ["station", "week", "rating"]
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64").astype(int)
    else:
        out = df[[station_col, rating_col]].copy()
        out.columns = ["station", "rating"]

    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").clip(0, 5)
    if "week" in out.columns:
        out = out.dropna(subset=["station", "week", "rating"]).copy()
        out = out.groupby(["station", "week"], as_index=False)["rating"].mean()
    else:
        out = out.dropna(subset=["station", "rating"]).copy()
        out = out.groupby(["station"], as_index=False)["rating"].mean()
    return out


def _load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Case 1: long (station[,week], price)
    long_has_station = any(c.lower() in {"station","location","resort","mountain","site"} for c in df.columns)
    long_has_price   = any(c.lower() in {"price","lift_pass_price","lift_price","day_pass","lift_ticket","ticket_price","value"} for c in df.columns)
    if long_has_station and long_has_price:
        station_col = _pick_col(df, ["station", "location", "resort", "mountain", "site"])
        price_col   = _pick_col(df, ["price", "lift_pass_price", "lift_price", "day_pass", "lift_ticket", "ticket_price", "value"])
        if any(c.lower()=="week" for c in df.columns):
            week_col = _pick_col(df, ["week"])
            out = df[[station_col, week_col, price_col]].copy()
            out.columns = ["station", "week", "price"]
            out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64").astype(int)
            out = out.groupby(["station", "week"], as_index=False)["price"].mean()
        else:
            out = df[[station_col, price_col]].copy()
            out.columns = ["station", "price"]
            out = out.groupby(["station"], as_index=False)["price"].mean()

        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        out = out.dropna(subset=["station", "price"]).copy()
        return out

    # Case 2: one-row wide (resorts are columns, first column a label)
    cols = list(df.columns)
    if len(cols) >= 2:
        sub = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        dens = sub.notna().sum(axis=1)
        ridx = int(dens.idxmax())
        price_vals = pd.to_numeric(df.iloc[ridx, 1:], errors="coerce")
        stations = cols[1:]
        out = pd.DataFrame({"station": stations, "price": price_vals.values})
        out = out.dropna(subset=["price"]).copy()
        return out

    raise ValueError("Could not parse prices file.")


# ---------------- station ID mapping ----------------
def _load_station_map(path: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    # accept common names + 'train_station' typo
    id_col   = _pick_col(m, ["rain_station", "train_station", "station_id", "rain_id", "bom_id", "code", "id"])
    name_col = _pick_col(m, ["station", "location", "resort", "mountain", "name"])
    out = m[[id_col, name_col]].copy()
    out.columns = ["rain_station", "mapped_station"]
    out["rain_station"] = out["rain_station"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["mapped_station"] = out["mapped_station"].astype(str).str.strip()
    return out


# ---------------- weighting ----------------
def _apply_weights(df: pd.DataFrame, w_rain: float, w_vis: float, w_rate: float, w_cost: float) -> pd.DataFrame:
    w = np.array([w_rain, w_vis, w_rate, w_cost], dtype=float)
    if np.any(w < 0) or np.all(w == 0):
        raise ValueError("Weights must be non-negative and not all zero.")
    w = w / w.sum()

    df = df.copy()
    df["rain_norm"]   = _minmax(df["rain_score"])
    df["vis_norm"]    = _minmax(df["visitor_score"])
    df["rating_norm"] = (pd.to_numeric(df["rating"], errors="coerce") / 5.0).clip(0, 1)
    df["cost_norm"]   = 1.0 - _minmax(df["price"])  # cheaper = better

    df["overall_score"] = (
        w[0] * df["rain_norm"] +
        w[1] * df["vis_norm"] +
        w[2] * df["rating_norm"] +
        w[3] * df["cost_norm"]
    )
    return df


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Combine snow/visitor/rating/cost into one weighted score + segmented picks.")
    ap.add_argument("--rain", required=True)
    ap.add_argument("--visitors", required=True)
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--target-week", type=int, default=None)
    ap.add_argument("--w-rain", type=float, default=0.5)
    ap.add_argument("--w-vis", type=float, default=0.3)
    ap.add_argument("--w-rate", type=float, default=0.1)
    ap.add_argument("--w-cost", type=float, default=0.1)
    ap.add_argument("--cheap-quantile", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="combined_scores.csv")
    ap.add_argument("--station-map", default=None,
                    help="CSV mapping from rain station IDs to resort names. Columns like: rain_station, station")
    ap.add_argument("--write-station-map-template", default=None,
                    help="If set and mapping is missing, write a template CSV of rain IDs to this path and exit.")
    ap.add_argument("--rain-agg", default="mean", help="Aggregator for duplicate rain rows per (station,week): mean|median|max")
    ap.add_argument("--vis-agg", default="mean", help="Aggregator for duplicate visitor rows per (station,week): mean|median|max")
    ap.add_argument("--debug", action="store_true", help="Print join-key diagnostics.")
    args = ap.parse_args()

    # Load inputs
    rain = _load_rain(args.rain)
    vis  = _load_visitors(args.visitors)
    rat  = _load_ratings(args.ratings)
    pri  = _load_prices(args.prices)

    # Map station IDs -> names if needed
    if _is_mostly_numeric(rain["station"]) and not _is_mostly_numeric(vis["station"]):
        rain_ids = rain["station"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        rain = rain.drop(columns=["station"]).assign(rain_station=rain_ids)

        if not args.station_map:
            out_tmpl = args.write_station_map_template or "station_map_template.csv"
            pd.DataFrame({"rain_station": sorted(rain["rain_station"].unique()), "station": ""}).to_csv(out_tmpl, index=False)
            raise SystemExit(
                f"\nCreated mapping template: {out_tmpl}\n"
                "Fill the 'station' names (e.g., 'Mt. Hotham', 'Falls Creek', ...), "
                "then re-run with: --station-map <that file>."
            )

        mapdf = _load_station_map(args.station_map)
        rain = rain.merge(mapdf, on="rain_station", how="left")
        if rain["mapped_station"].isna().any():
            missing_ids = sorted(rain.loc[rain["mapped_station"].isna(), "rain_station"].unique())
            raise ValueError(f"The following rain station IDs are missing in {args.station_map}: {missing_ids}")
        rain = rain.assign(station=rain["mapped_station"]).drop(columns=["rain_station", "mapped_station"])

    # Add normalized keys for join
    rain = _add_key(rain, "station")
    vis  = _add_key(vis,  "station")
    rat  = _add_key(rat,  "station")
    pri  = _add_key(pri,  "station")

    # Collapse duplicates per (station_key, week)
    rain_before = len(rain); vis_before = len(vis)
    rain = _collapse_duplicates(
        rain, keys=["station_key", "week"],
        numeric_aggs={"rain_score": _agg_name_to_func(args.rain_agg)},
        prefer_first_cols=["station"]
    )
    vis = _collapse_duplicates(
        vis, keys=["station_key", "week"],
        numeric_aggs={"visitor_score": _agg_name_to_func(args.vis_agg)},
        prefer_first_cols=["station"]
    )
    if len(rain) < rain_before:
        print(f"[info] Collapsed duplicate rain rows: {rain_before - len(rain)}")
    if len(vis) < vis_before:
        print(f"[info] Collapsed duplicate visitor rows: {vis_before - len(vis)}")

    # Also dedupe ratings/prices to be safe
    if "week" in rat.columns:
        rat = _collapse_duplicates(rat, ["station_key", "week"], {"rating": "mean"}, ["station"])
    else:
        rat = _collapse_duplicates(rat, ["station_key"], {"rating": "mean"}, ["station"])
    if "week" in pri.columns:
        pri = _collapse_duplicates(pri, ["station_key", "week"], {"price": "mean"}, ["station"])
    else:
        pri = _collapse_duplicates(pri, ["station_key"], {"price": "mean"}, ["station"])

    # Debug sets
    if args.debug:
        print("[debug] rain keys (sample):", rain["station_key"].unique()[:10])
        print("[debug] vis  keys (sample):", vis["station_key"].unique()[:10])
        inter = set(rain["station_key"]).intersection(set(vis["station_key"]))
        print("[debug] intersect size:", len(inter))

    # Base join: rain ∩ visitors on station_key + week
    df = pd.merge(
        rain.rename(columns={"station": "station_rain"}),
        vis.rename(columns={"station": "station_vis"}),
        on=["station_key", "week"],
        how="inner"
    )

    # display name: prefer rain's, else visitors'
    df["station"] = df["station_rain"].where(
        df["station_rain"].notna() & (df["station_rain"].astype(str) != ""), df["station_vis"]
    )
    df = df.drop(columns=["station_rain", "station_vis"])

    # Join ratings (prefer week-aware if present)
    if "week" in rat.columns:
        df = pd.merge(df, rat[["station_key", "week", "rating"]], on=["station_key", "week"], how="left")
    else:
        df = pd.merge(df, rat[["station_key", "rating"]], on="station_key", how="left")

    # Join prices (prefer week-aware if present)
    if "week" in pri.columns:
        df = pd.merge(df, pri[["station_key", "week", "price"]], on=["station_key", "week"], how="left")
    else:
        df = pd.merge(df, pri[["station_key", "price"]], on="station_key", how="left")

    # Target week filter
    if args.target_week is not None:
        df = df[df["week"].astype(int) == int(args.target_week)].copy()

    if df.empty:
        raise ValueError("No overlapping (station, week) rows after joins. Check station mapping and names.")

    # Fill missing rating/price with neutral values
    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(2.5)
    if df["price"].isna().any():
        df["price"] = df["price"].fillna(df["price"].median())

    # Weights + segment
    df = _apply_weights(df, args.w_rain, args.w_vis, args.w_rate, args.w_cost)
    cut = df["price"].quantile(args.cheap_quantile)
    df["segment"] = np.where(df["price"] <= cut, "CHEAP", "EXPENSIVE")

    # Rank and print
    df = df.sort_values(["overall_score", "price"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    preview_cols = ["rank", "station", "week", "price", "rating", "rain_score", "visitor_score",
                    "rain_norm", "vis_norm", "rating_norm", "cost_norm", "overall_score", "segment"]
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.precision", 3):
        print(df[preview_cols].head(args.top).to_string(index=False))

    # Save
    df.to_csv(args.out, index=False)
    print(f"\nSaved full combined ranking to: {args.out}")

    # Picks
    best_overall = df.iloc[0]
    cheap_df = df[df["segment"] == "CHEAP"]
    rich_df  = df[df["segment"] == "EXPENSIVE"]
    best_budget  = cheap_df.iloc[0] if not cheap_df.empty else best_overall
    best_wealthy = rich_df.iloc[0]  if not rich_df.empty  else best_overall

    def fmt(row, label):
        return (f"{label} → {row.station}, week {int(row.week)} | overall {row.overall_score:.3f} "
                f"| price ${row.price:,.0f} | rating {row.rating:.1f}/5 | "
                f"rain {row.rain_score:.3f} | visitors {row.visitor_score:.3f}")

    print("\nRecommendations")
    print(fmt(best_overall, "Neutral (overall best)"))
    print(fmt(best_budget,  "Budget family (CHEAP)"))
    print(fmt(best_wealthy, "Wealthy family (EXPENSIVE)"))


if __name__ == "__main__":
    main()
