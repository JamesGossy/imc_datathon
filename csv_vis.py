import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


vis = pd.read_csv("data/visitation_data.csv")
clim = pd.read_csv("data/climate_data.csv")

def _find_col(cols, target):
    for c in cols:
        if str(c).strip().lower() == target:
            return c
    raise KeyError(f"Missing required column '{target}' (case-insensitive).")

YEAR = _find_col(vis.columns, "year")
WEEK = _find_col(vis.columns, "week")
id_cols = [YEAR, WEEK]
resort_cols = [c for c in vis.columns if c not in id_cols]

# ensure numeric Year/Week
vis[YEAR] = pd.to_numeric(vis[YEAR], errors="coerce").astype("Int64")
vis[WEEK] = pd.to_numeric(vis[WEEK], errors="coerce").astype("Int64")

vis[resort_cols] = vis[resort_cols].apply(pd.to_numeric, errors="coerce")
vis[resort_cols] = vis[resort_cols].fillna(vis.groupby(WEEK)[resort_cols].transform("median"))
vis[resort_cols] = vis[resort_cols].fillna(vis[resort_cols].median())
vis = vis[~vis[YEAR].isin([2020, 2021])].copy()
vis.to_csv("data/visitation_data_imputed.csv", index=False)

clim = clim.loc[clim["Year"] >= 2014].copy()
mask = (
    ((clim["Month"] == 6) & (clim["Day"] >= 9)) |
    (clim["Month"].isin([7, 8])) |
    ((clim["Month"] == 9) & (clim["Day"] <= 21))
)
clim = clim.loc[mask].copy()

d = pd.to_datetime(dict(year=2000, month=clim["Month"], day=clim["Day"]))
clim["Week"] = ((d - pd.Timestamp(2000, 6, 9)).dt.days // 7 + 1).astype(int)

clim_cols = [
    "Maximum temperature (Degree C)",
    "Minimum temperature (Degree C)",
    "Rainfall amount (millimetres)",
]
week_av_clim_df = (
    clim.groupby(["Bureau of Meteorology station number", "Year", "Week"])[clim_cols]
        .median()
        .reset_index()
)
fill_vals = week_av_clim_df.groupby(["Bureau of Meteorology station number", "Week"])[clim_cols].transform("median")
week_av_clim_df[clim_cols] = week_av_clim_df[clim_cols].fillna(fill_vals)
week_av_clim_df.to_csv("data/climate_data_filtered_imputed.csv", index=False)

resort_cols_clean = [c for c in vis.columns if c not in id_cols]

long = vis.melt(id_vars=id_cols, value_vars=resort_cols_clean,
                var_name="Resort", value_name="Visitation")

weekly_resort_avg = (
    long.groupby([WEEK, "Resort"], as_index=False)["Visitation"]
        .mean()
        .sort_values(["Resort", WEEK])
)

vis["Total_Visitation"] = vis[resort_cols_clean].sum(axis=1)
weekly_total_avg = (
    vis.groupby(WEEK, as_index=False)["Total_Visitation"]
       .mean()
       .sort_values(WEEK)
)

yearly_resort_total = (
    long.groupby([YEAR, "Resort"], as_index=False)["Visitation"]
        .sum()
        .sort_values(["Resort", YEAR])
)

yearly_total = (
    vis.groupby(YEAR)[resort_cols_clean].sum().sum(axis=1)
       .reset_index(name="Total_Visitation")
       .sort_values(YEAR)
)

snow = long.dropna(subset=[YEAR, WEEK, "Visitation"]).copy()
years_ok = [y for y in range(2015, 2025) if y not in (2020, 2021)]
snow = snow[snow[YEAR].isin(years_ok) & snow[WEEK].between(1, 15)].copy()
snow = snow.sort_values(["Resort", YEAR, WEEK])
snow["Week_Idx"] = snow.groupby("Resort").cumcount() + 1
snow["Resort"] = pd.Categorical(snow["Resort"], categories=resort_cols_clean, ordered=True)

n_years = snow[YEAR].nunique()
n_weeks = snow[WEEK].nunique()
max_pts = n_years * n_weeks
min_year, max_year = snow[YEAR].min(), snow[YEAR].max() 

snow_avg = (
    snow.groupby("Week_Idx", as_index=False)["Visitation"]
        .mean()
        .rename(columns={"Visitation": "Avg_Visitation"})
)

# 1) Average visitation per resort per week
plt.figure(figsize=(12, 7))
sns.lineplot(data=weekly_resort_avg, x=WEEK, y="Visitation", hue="Resort", marker="o")
plt.title("Average Visitation per Resort per Week")
plt.grid(True); plt.tight_layout(); plt.show()

# 2) Average total visitation per week
plt.figure(figsize=(12, 7))
sns.lineplot(data=weekly_total_avg, x=WEEK, y="Total_Visitation", marker="o")
plt.title("Average Total Visitation per Week (All Resorts)")
plt.grid(True); plt.tight_layout(); plt.show()

# 3) Yearly visitation by resort
plt.figure(figsize=(12, 7))
sns.lineplot(data=yearly_resort_total, x=YEAR, y="Visitation", hue="Resort", marker="o")
plt.title("Yearly Visitation by Resort")
plt.grid(True); plt.tight_layout(); plt.show()

# 4) Consecutive snow-season weeks per resort (2015–2019, 2022–2024; weeks 1–15)
plt.figure(figsize=(12, 7))
sns.lineplot(data=snow, x="Week_Idx", y="Visitation", hue="Resort", marker="o")
plt.title(
    f"Consecutive Snow-Season Weeks per Resort ({min_year}–{max_year})\n"
    f"{n_weeks} weeks × {n_years} years = up to {max_pts} points per line"
)
plt.xlabel("Consecutive Week Index")
plt.ylabel("Visitors")
plt.grid(True); plt.tight_layout(); plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=snow_avg, x="Week_Idx", y="Avg_Visitation", marker="o")
plt.title(f"Average Visitation Across All Resorts by Consecutive Week ({min_year}–{max_year})")
plt.xlabel("Consecutive Week Index")
plt.ylabel("Average Visitors")
plt.grid(True); plt.tight_layout(); plt.show()

from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from darts.models import (NaiveSeasonal, ExponentialSmoothing, Theta, FFT, LinearRegressionModel, AutoARIMA)
from darts.utils.utils import SeasonalityMode, ModelMode

# --- Helper: aggregate to weekly mean across all resorts; keep weeks 1..15
hist = (
    long.dropna(subset=[YEAR, WEEK, "Visitation"])
        .loc[long[WEEK].between(1, 15)]
        .groupby([YEAR, WEEK], as_index=False)["Visitation"].mean()
        .rename(columns={"Visitation": "MeanVisitors"})
        .sort_values([YEAR, WEEK])
)

# Holdout year (use 2025 if present, else the latest)
years = pd.Index(sorted(hist[YEAR].unique()))
eval_year = 2025 if 2025 in years else int(years.max())

train_df = hist[hist[YEAR] < eval_year].sort_values([YEAR, WEEK])
test_df  = hist[hist[YEAR] == eval_year].sort_values(WEEK)

# Make evenly spaced 7-day timeline for Darts
def to_ts(df):
    idx = pd.date_range("2000-06-09", periods=len(df), freq="7D")
    return TimeSeries.from_times_and_values(idx, df["MeanVisitors"].to_numpy())

train_ts = to_ts(train_df)
test_ts  = to_ts(test_df)

# --- Eval/plot helper
import matplotlib.pyplot as plt

def eval_model(name, forecast_ts, plot=False):
    y_true = test_ts.values().ravel()
    y_pred = forecast_ts.values().ravel()
    rmse = mean_squared_error(y_true, y_pred)
    print(f"{eval_year} {name} RMSE: {rmse:.2f}  [n={len(y_true)}]")
    if plot:
        weeks = test_df[WEEK].to_numpy()
        plt.figure(figsize=(8, 4.5))
        plt.plot(weeks, y_true, marker='o', label=f'Actual {eval_year}')
        plt.plot(weeks, y_pred, marker='o', linestyle='--', label=name)
        plt.xlabel('Week'); plt.ylabel('Mean visitors')
        plt.title(f'{name}'); plt.legend()
        plt.grid(True); plt.tight_layout(); plt.show()
    return rmse

scores = {}

# 1) Seasonal Naive (K = 15 weeks)
m_ns = NaiveSeasonal(K=15)
m_ns.fit(train_ts)
fc_ns = m_ns.predict(len(test_ts))
scores["NaiveSeasonal(K=15)"] = eval_model("NaiveSeasonal(K=15)", fc_ns, plot=True)

# 2) Exponential Smoothing (additive seasonality, no trend)
es = ExponentialSmoothing(
    seasonal=SeasonalityMode.ADDITIVE,
    seasonal_periods=15,
    trend=ModelMode.NONE
)
es.fit(train_ts)
fc_es = es.predict(len(test_ts))
scores["ExpSmoothing(15, add)"] = eval_model("ExpSmoothing(15, add)", fc_es, plot=True)

# 3) Theta (strong simple classic)
m_theta = Theta(seasonality_period=15, season_mode=SeasonalityMode.ADDITIVE)
m_theta.fit(train_ts)
fc_theta = m_theta.predict(len(test_ts))
scores["Theta(15, add)"] = eval_model("Theta(15, add)", fc_theta, plot=True)

# 4) FFT (Fourier; keep a few dominant frequencies)
m_fft = FFT(nr_freqs_to_keep=5)
m_fft.fit(train_ts)
fc_fft = m_fft.predict(len(test_ts))
scores["FFT(k=5)"] = eval_model("FFT(k=5)", fc_fft, plot=True)

# 5) Linear Regression on lags (uses last 15 weeks as features)
m_lin = LinearRegressionModel(lags=15)
m_lin.fit(train_ts)
fc_lin = m_lin.predict(len(test_ts))
scores["Linear(lags=15)"] = eval_model("Linear(lags=15)", fc_lin, plot=True)

# 6) AutoARIMA if available; skip silently if pmdarima not installed
m_aari = AutoARIMA(season_length=15)
m_aari.fit(train_ts)
fc_aari = m_aari.predict(len(test_ts))
scores["AutoARIMA(m=15)"] = eval_model("AutoARIMA(m=15)", fc_aari, plot=True)

# --- Print leaderboard
print("\nRMSE leaderboard:")
for name, s in sorted(scores.items(), key=lambda kv: kv[1]):
    print(f"{name:<22} {s:.2f}")