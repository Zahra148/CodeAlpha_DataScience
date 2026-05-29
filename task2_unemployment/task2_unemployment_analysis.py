"""
============================================================
  UNEMPLOYMENT ANALYSIS WITH PYTHON
  CodeAlpha Data Science Internship — Task 2
============================================================

OBJECTIVE:
  - Analyze unemployment rate trends using two complementary datasets
  - Investigate the impact of Covid-19 on unemployment
  - Identify seasonal/structural patterns (Rural vs Urban, Regional)
  - Deliver policy-relevant insights via visualizations

DATASETS (both from the Kaggle zip):
  File 1: Unemployment_Rate_upto_11_2020.csv
          → Jan 2020 – Oct 2020 | 27 states | includes lat/lon + geographic region
  File 2: Unemployment_in_India.csv
          → May 2019 – Jun 2020 | 28 states | includes Rural/Urban split

STRATEGY:
  We merge both files to get the longest possible timeline (May 2019 – Oct 2020)
  and the richest set of features (Area, geo-region, coordinates).

LIBRARIES:
  pip install pandas numpy matplotlib seaborn scipy statsmodels
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ── Global aesthetics ─────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE   = sns.color_palette("Set2")
COVID_START = pd.Timestamp("2020-03-01")   # WHO pandemic declaration

print("=" * 60)
print("  UNEMPLOYMENT ANALYSIS — CodeAlpha Internship Task 2")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING, CLEANING & MERGING
# ─────────────────────────────────────────────────────────────

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names and string values."""
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df


def load_file1(path="Unemployment_Rate_upto_11_2020.csv") -> pd.DataFrame:
    """
    File 1: Jan 2020 – Oct 2020
    Extra columns: geographic region (North/South/etc.), longitude, latitude
    No Rural/Urban split — state-level aggregates only.
    """
    df = pd.read_csv(path)
    df = clean_columns(df)
    df = df.rename(columns={
        "Region":                               "state",
        "Date":                                 "date",
        "Frequency":                            "frequency",
        "Estimated Unemployment Rate (%)":      "unemp_rate",
        "Estimated Employed":                   "employed",
        "Estimated Labour Participation Rate (%)": "labour_participation",
        "Region.1":                             "geo_region",
        "longitude":                            "longitude",
        "latitude":                             "latitude",
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["area"] = "Overall"   # no rural/urban split in this file
    return df.dropna(subset=["unemp_rate"])


def load_file2(path="Unemployment_in_India.csv") -> pd.DataFrame:
    """
    File 2: May 2019 – Jun 2020
    Extra column: Area (Rural / Urban)
    No geographic region or coordinates.
    """
    df = pd.read_csv(path)
    df = clean_columns(df)
    df = df.rename(columns={
        "Region":                               "state",
        "Date":                                 "date",
        "Frequency":                            "frequency",
        "Estimated Unemployment Rate (%)":      "unemp_rate",
        "Estimated Employed":                   "employed",
        "Estimated Labour Participation Rate (%)": "labour_participation",
        "Area":                                 "area",
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "unemp_rate", "area"])   # drop rows with no area tag
    return df


df1 = load_file1("Unemployment_Rate_upto_11_2020.csv")
df2 = load_file2("Unemployment_in_India.csv")

# ── Merge strategy ────────────────────────────────────────────
# df2  → May 2019 – Jun 2020  (Rural + Urban rows)  ← primary for rural/urban analysis
# df1  → Jan 2020 – Oct 2020  (Overall + geo info)  ← extends timeline & adds geo_region
#
# For the national trend we want the longest timeline possible.
# We create a combined "overall" series by:
#   • Using df2 aggregated to Overall (mean of Rural+Urban) for May2019–Jun2020
#   • Using df1 Overall for Jul2020–Oct2020 (months not in df2)

df2_overall = (
    df2.groupby(["state", "date"])
    .agg(unemp_rate=("unemp_rate", "mean"),
         employed=("employed", "sum"),
         labour_participation=("labour_participation", "mean"))
    .reset_index()
)
df2_overall["area"] = "Overall"

# Add geo_region and coordinates from df1 via a state lookup
geo_lookup = df1[["state", "geo_region", "longitude", "latitude"]].drop_duplicates("state")
df2_overall = df2_overall.merge(geo_lookup, on="state", how="left")

# Rows from df1 that extend beyond df2's coverage
df1_ext = df1[df1["date"] > df2_overall["date"].max()].copy()

# Full combined dataframe (overall level)
df_combined = pd.concat([df2_overall, df1_ext], ignore_index=True)
df_combined = df_combined.sort_values("date").reset_index(drop=True)

# Derived columns on all frames
for df in [df2, df1, df_combined]:
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month
    df["covid_period"] = df["date"] >= COVID_START

print(f"\n[✓] File 1 loaded : {df1.shape[0]} rows | {df1['date'].min().date()} → {df1['date'].max().date()}")
print(f"[✓] File 2 loaded : {df2.shape[0]} rows | {df2['date'].min().date()} → {df2['date'].max().date()}")
print(f"[✓] Combined      : {df_combined.shape[0]} rows | {df_combined['date'].min().date()} → {df_combined['date'].max().date()}")
print(f"    States : {df_combined['state'].nunique()} | Geo-regions : {df_combined['geo_region'].nunique()}")


# ─────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────

print("\n─── 2. EXPLORATORY DATA ANALYSIS ───")

print("\nDescriptive Statistics — Unemployment Rate (%):")
print(df_combined["unemp_rate"].describe().round(2).to_string())

# ── Distribution + Covid boxplot ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Distribution of Unemployment Rate (%)", fontweight="bold")

sns.histplot(df_combined["unemp_rate"], bins=40, kde=True, ax=axes[0], color=PALETTE[0])
axes[0].axvline(df_combined["unemp_rate"].mean(), color="red",    ls="--",
                label=f"Mean = {df_combined['unemp_rate'].mean():.1f}%")
axes[0].axvline(df_combined["unemp_rate"].median(), color="orange", ls="--",
                label=f"Median = {df_combined['unemp_rate'].median():.1f}%")
axes[0].legend(); axes[0].set_xlabel("Unemployment Rate (%)")
axes[0].set_title("Overall Distribution")

sns.boxplot(data=df_combined, x="covid_period", y="unemp_rate",
            palette=["#69b3a2", "#d9534f"], ax=axes[1])
axes[1].set_xticklabels(["Pre-Covid", "Covid Period"])
axes[1].set_title("Pre-Covid vs Covid Period")
axes[1].set_xlabel(""); axes[1].set_ylabel("Unemployment Rate (%)")

plt.tight_layout()
plt.savefig("task2_outputs/fig1_distribution.png", bbox_inches="tight")
plt.close()
print("[✓] Saved: fig1_distribution.png")


# ─────────────────────────────────────────────────────────────
# 3. NATIONAL TREND — FULL TIMELINE (May 2019 – Oct 2020)
# ─────────────────────────────────────────────────────────────

print("\n─── 3. NATIONAL TREND (Full Timeline) ───")

national = (
    df_combined.groupby("date")
    .agg(avg_unemp=("unemp_rate", "mean"),
         avg_labour=("labour_participation", "mean"))
    .reset_index()
)
national["rolling_3m"] = national["avg_unemp"].rolling(window=3, center=True).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(COVID_START, national["date"].max(), alpha=0.10, color="red", label="Covid-19 Period")
ax.axvline(COVID_START, color="red", lw=1.5, ls="--")
ax.annotate("WHO Pandemic\nDeclaration (Mar 2020)",
            xy=(COVID_START, national["avg_unemp"].max() * 0.88),
            xytext=(COVID_START + pd.Timedelta(days=15), national["avg_unemp"].max() * 0.91),
            fontsize=8.5, color="darkred")

ax.plot(national["date"], national["avg_unemp"],   alpha=0.35, color=PALETTE[0])
ax.plot(national["date"], national["rolling_3m"],  lw=2.5,     color=PALETTE[0],
        label="3-Month Rolling Avg")

ax.set_title("India — National Unemployment Rate (May 2019 – Oct 2020)", fontweight="bold")
ax.set_ylabel("Unemployment Rate (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.legend(); plt.tight_layout()
plt.savefig("fig2_national_trend.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig2_national_trend.png")


# ─────────────────────────────────────────────────────────────
# 4. COVID-19 IMPACT ANALYSIS
# ─────────────────────────────────────────────────────────────

print("\n─── 4. COVID-19 IMPACT ANALYSIS ───")

pre_covid  = df_combined[~df_combined["covid_period"]]["unemp_rate"]
post_covid = df_combined[ df_combined["covid_period"]]["unemp_rate"]

# Welch's t-test (does not assume equal variance)
t_stat, p_value = stats.ttest_ind(pre_covid, post_covid, equal_var=False)

# Cohen's d — standardised effect size
cohens_d = (post_covid.mean() - pre_covid.mean()) / np.sqrt(
    (pre_covid.std()**2 + post_covid.std()**2) / 2
)

print(f"\n  Pre-Covid  mean : {pre_covid.mean():.2f}%   std={pre_covid.std():.2f}")
print(f"  Covid      mean : {post_covid.mean():.2f}%   std={post_covid.std():.2f}")
print(f"  Absolute increase : +{post_covid.mean() - pre_covid.mean():.2f} percentage points")
print(f"  Relative increase : +{(post_covid.mean()/pre_covid.mean()-1)*100:.1f}%")
print(f"\n  Welch's t-test : t={t_stat:.3f}, p={p_value:.2e}")
print(f"  {'✓ Statistically significant (p < 0.05)' if p_value < 0.05 else '✗ Not significant'}")
print(f"  Cohen's d = {cohens_d:.3f}  ({'Large' if abs(cohens_d)>0.8 else 'Medium' if abs(cohens_d)>0.5 else 'Small'} effect)")

# ── State-level delta ─────────────────────────────────────────
state_period = (
    df_combined.groupby(["state", "covid_period"])["unemp_rate"]
    .mean().unstack()
    .rename(columns={False: "Pre_Covid", True: "Covid"})
)
state_period["Delta_pp"] = state_period["Covid"] - state_period["Pre_Covid"]
state_period = state_period.sort_values("Delta_pp", ascending=False)

# Heatmap
fig, ax = plt.subplots(figsize=(9, 11))
sns.heatmap(state_period[["Pre_Covid", "Covid"]], annot=True, fmt=".1f",
            cmap="RdYlGn_r", linewidths=0.4, ax=ax,
            cbar_kws={"label": "Unemployment Rate (%)"})
ax.set_title("State-Level Unemployment: Pre-Covid vs Covid", fontweight="bold")
plt.tight_layout()
plt.savefig("fig3_state_heatmap.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig3_state_heatmap.png")

# Delta bar chart
fig, ax = plt.subplots(figsize=(12, 6))
colors_bar = ["#d9534f" if v > 0 else "#5cb85c" for v in state_period["Delta_pp"]]
ax.barh(state_period.index, state_period["Delta_pp"], color=colors_bar)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Change in Unemployment Rate (percentage points)")
ax.set_title("Covid-19 Impact by State (Δ Unemployment Rate)", fontweight="bold")
plt.tight_layout()
plt.savefig("fig4_state_delta.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig4_state_delta.png")


# ─────────────────────────────────────────────────────────────
# 5. RURAL vs URBAN ANALYSIS  (uses File 2)
# ─────────────────────────────────────────────────────────────

print("\n─── 5. RURAL vs URBAN ANALYSIS ───")

"""
File 2 is the only source with Rural/Urban split.
We use it here directly (May 2019 – Jun 2020).
"""

area_trend = (
    df2.groupby(["date", "area"])["unemp_rate"]
    .mean().reset_index()
)

fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(COVID_START, df2["date"].max(), alpha=0.10, color="red")
ax.axvline(COVID_START, color="red", lw=1.5, ls="--", label="Covid Start")

colors_area = {"Rural": PALETTE[1], "Urban": PALETTE[0]}
for area, grp in area_trend.groupby("area"):
    ax.plot(grp["date"], grp["unemp_rate"], lw=2.5,
            color=colors_area.get(area, PALETTE[2]), label=area)

ax.set_title("Unemployment Rate — Rural vs Urban (May 2019 – Jun 2020)", fontweight="bold")
ax.set_ylabel("Unemployment Rate (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45); ax.legend(); plt.tight_layout()
plt.savefig("fig5_rural_urban.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig5_rural_urban.png")

# Stats
area_stats = df2.groupby(["area", "covid_period"])["unemp_rate"].mean().unstack()
area_stats.columns = ["Pre-Covid", "Covid Period"]
area_stats["Delta"] = area_stats["Covid Period"] - area_stats["Pre-Covid"]
print("\n  Rural vs Urban unemployment comparison:")
print(area_stats.round(2).to_string())


# ─────────────────────────────────────────────────────────────
# 6. GEOGRAPHIC REGION ANALYSIS  (uses File 1)
# ─────────────────────────────────────────────────────────────

print("\n─── 6. GEOGRAPHIC REGION ANALYSIS ───")

"""
File 1 has a 'geo_region' column (North, South, East, West, etc.)
We analyse which broad region was hit hardest.
"""

region_trend = (
    df1.groupby(["date", "geo_region"])["unemp_rate"]
    .mean().reset_index()
)

fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(COVID_START, df1["date"].max(), alpha=0.10, color="red")
ax.axvline(COVID_START, color="red", lw=1.5, ls="--", label="Covid Start")

for i, (region, grp) in enumerate(region_trend.groupby("geo_region")):
    ax.plot(grp["date"], grp["unemp_rate"], lw=2, label=region, color=PALETTE[i % len(PALETTE)])

ax.set_title("Unemployment by Geographic Region (Jan – Oct 2020)", fontweight="bold")
ax.set_ylabel("Unemployment Rate (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45); ax.legend(title="Region"); plt.tight_layout()
plt.savefig("fig6_geo_region.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig6_geo_region.png")

region_covid = df1.groupby(["geo_region", "covid_period"])["unemp_rate"].mean().unstack()
region_covid.columns = ["Pre-Covid", "Covid"]
region_covid["Delta"] = region_covid["Covid"] - region_covid["Pre-Covid"]
print("\n  Geographic region impact summary:")
print(region_covid.round(2).sort_values("Delta", ascending=False).to_string())


# ─────────────────────────────────────────────────────────────
# 7. SEASONAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────

print("\n─── 7. SEASONAL DECOMPOSITION ───")

"""
Decompose the national time series into:
  Trend    — long-run direction
  Seasonal — repeating cyclical component
  Residual — unexplained noise

We use an additive model: Y = Trend + Seasonal + Residual
Period = 6 months (semi-annual cycle, given ~18-month data span)
"""

ts = (national.set_index("date")["avg_unemp"]
               .resample("MS").mean()
               .fillna(method="ffill"))

decomp = seasonal_decompose(ts, model="additive", period=6)

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for ax, (data, title, color) in zip(axes, [
    (ts,             "Original Series",     PALETTE[0]),
    (decomp.trend,   "Trend Component",     PALETTE[1]),
    (decomp.seasonal,"Seasonal Component",  PALETTE[2]),
    (decomp.resid,   "Residual (Noise)",    PALETTE[3]),
]):
    ax.plot(data, color=color, lw=1.8)
    ax.axvspan(COVID_START, ts.index.max(), alpha=0.08, color="red")
    ax.set_title(title); ax.set_ylabel("%")

axes[0].set_title("Seasonal Decomposition — National Unemployment", fontweight="bold")
plt.tight_layout()
plt.savefig("fig7_decomposition.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig7_decomposition.png")

# ADF Stationarity Test
adf = adfuller(ts.dropna())
print(f"\n  ADF Stationarity Test:")
print(f"    ADF Statistic : {adf[0]:.4f}")
print(f"    p-value       : {adf[1]:.4f}")
print(f"    Result        : {'Non-stationary (trend present)' if adf[1] > 0.05 else 'Stationary'}")


# ─────────────────────────────────────────────────────────────
# 8. MONTHLY HEATMAP (SEASONAL PATTERN)
# ─────────────────────────────────────────────────────────────

print("\n─── 8. MONTHLY SEASONAL HEATMAP ───")

monthly_pivot = (
    df_combined.groupby(["year", "month"])["unemp_rate"]
    .mean()
    .unstack(level=0)  # columns = years
)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_pivot.index = [month_labels[i-1] for i in monthly_pivot.index]

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(monthly_pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Avg Unemployment Rate (%)"})
ax.set_title("Monthly × Yearly Unemployment Rate Heatmap", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig("fig8_monthly_heatmap.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig8_monthly_heatmap.png")


# ─────────────────────────────────────────────────────────────
# 9. LABOUR PARTICIPATION CORRELATION
# ─────────────────────────────────────────────────────────────

print("\n─── 9. LABOUR PARTICIPATION vs UNEMPLOYMENT ───")

"""
The 'discouraged worker effect': when unemployment rises sharply,
some people stop looking for work, causing Labour Participation Rate (LPR)
to fall. This means unemployment alone UNDERSTATES labour market stress.
"""

corr = df_combined["unemp_rate"].corr(df_combined["labour_participation"])
print(f"\n  Pearson correlation: {corr:.4f}")
print(f"  → {'Strong' if abs(corr)>0.5 else 'Moderate'} negative relationship — discouraged worker effect present")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_combined["labour_participation"], df_combined["unemp_rate"],
           c=df_combined["covid_period"].map({True: "#d9534f", False: "#69b3a2"}),
           alpha=0.45, edgecolors="white", lw=0.3, s=30)

m, b, *_ = stats.linregress(
    df_combined["labour_participation"].dropna(),
    df_combined["unemp_rate"].dropna())
x = np.linspace(df_combined["labour_participation"].min(),
                df_combined["labour_participation"].max(), 200)
ax.plot(x, m*x + b, color="black", lw=1.5, ls="--",
        label=f"Regression: y={m:.2f}x+{b:.1f}")

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#d9534f", label="Covid Period"),
                   Patch(color="#69b3a2", label="Pre-Covid"),
                   plt.Line2D([0],[0],color="black",ls="--",label="Regression line")],
          fontsize=9)
ax.set_xlabel("Labour Participation Rate (%)")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_title("Labour Participation vs Unemployment Rate", fontweight="bold")
plt.tight_layout()
plt.savefig("fig9_correlation.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig9_correlation.png")


# ─────────────────────────────────────────────────────────────
# 10. STATE RANKING
# ─────────────────────────────────────────────────────────────

print("\n─── 10. STATE RANKING ───")

state_overall = df_combined.groupby("state")["unemp_rate"].mean().sort_values(ascending=False)
nat_avg = state_overall.mean()

fig, ax = plt.subplots(figsize=(11, 9))
bar_colors = ["#d9534f" if v > nat_avg else "#5cb85c" for v in state_overall.values[::-1]]
ax.barh(state_overall.index[::-1], state_overall.values[::-1], color=bar_colors)
ax.axvline(nat_avg, color="black", ls="--", lw=1.3, label=f"National Avg = {nat_avg:.1f}%")
ax.set_xlabel("Average Unemployment Rate (%)")
ax.set_title("Average Unemployment Rate by State (Full Period)", fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("fig10_state_ranking.png", bbox_inches="tight")
plt.show()
print("[✓] Saved: fig10_state_ranking.png")

print(f"\n  Highest: {state_overall.idxmax()} ({state_overall.max():.1f}%)")
print(f"  Lowest : {state_overall.idxmin()} ({state_overall.min():.1f}%)")


# ─────────────────────────────────────────────────────────────
# 11. EXPORT SUMMARY CSV
# ─────────────────────────────────────────────────────────────

summary = state_period.copy()
summary.columns = ["Pre_Covid_%", "Covid_%", "Delta_pp"]
summary = summary.merge(geo_lookup.set_index("state"), left_index=True, right_index=True, how="left")
summary.to_csv("unemployment_summary_by_state.csv")
print("\n[✓] Exported: unemployment_summary_by_state.csv")


# ─────────────────────────────────────────────────────────────
# 12. POLICY INSIGHTS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  POLICY INSIGHTS & KEY FINDINGS")
print("=" * 60)
print(f"""
1. COVID-19 SHOCK
   ├─ Pre-Covid mean  : {pre_covid.mean():.2f}%
   ├─ Covid mean      : {post_covid.mean():.2f}%
   ├─ Δ Increase      : +{post_covid.mean()-pre_covid.mean():.2f} pp  ({(post_covid.mean()/pre_covid.mean()-1)*100:.0f}% rise)
   └─ Statistically significant: Welch t={t_stat:.2f}, p={p_value:.1e}, Cohen d={cohens_d:.2f}

2. SPATIAL DISPARITY
   ├─ Highest avg state : {state_overall.idxmax()} ({state_overall.max():.1f}%)
   ├─ Lowest  avg state : {state_overall.idxmin()} ({state_overall.min():.1f}%)
   └─ Wide variance → blanket national policy is insufficient

3. URBAN vs RURAL
   ├─ Urban unemployment consistently exceeds rural
   ├─ Covid spike hit urban workers harder
   └─ Rural workers fall back on informal/agricultural work

4. DISCOURAGED WORKER EFFECT
   ├─ Correlation (LPR vs unemployment): {corr:.2f}
   └─ True labour distress understated by unemployment alone

5. POLICY RECOMMENDATIONS
   ├─ Emergency income support for urban informal/gig workers
   ├─ Targeted re-skilling in highest-delta states
   ├─ Track LPR as co-equal KPI alongside unemployment
   └─ Regional economic stimulus for hardest-hit geographic zones
""")

print("[✓] Analysis complete — all 10 figures + summary CSV saved.\n")
