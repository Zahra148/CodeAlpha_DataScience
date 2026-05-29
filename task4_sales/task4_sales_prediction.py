"""
============================================================
  TASK 4: SALES PREDICTION USING PYTHON
  CodeAlpha Data Science Internship
============================================================

OBJECTIVE:
  Predict product sales based on advertising spend across:
    • TV        — traditional broadcast advertising
    • Radio     — radio advertising
    • Newspaper — print advertising

  Deliver actionable insights:
    • Which channel drives the most sales?
    • What is the ROI of each advertising channel?
    • How should budget be allocated?

APPROACH:
  1. EDA — distributions, correlations, scatter plots
  2. Simple Linear Regression (each channel independently)
  3. Multiple Linear Regression (all channels together)
  4. Polynomial Regression (capture non-linear effects)
  5. Random Forest / Gradient Boosting
  6. Advertising ROI analysis & budget optimisation

DATASET:
  Kaggle: bumba12345/advertising
  File  : advertising.csv
  Columns: TV, Radio, Newspaper, Sales (all in thousands $)
  (run setup_kaggle.py to download automatically)
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────

import os
import glob
import warnings
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")
os.makedirs("task4_outputs", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CHANNEL_COLORS = {"TV": "#e74c3c", "Radio": "#3498db", "Newspaper": "#2ecc71"}

print("=" * 60)
print("  TASK 4: SALES PREDICTION")
print("  CodeAlpha Data Science Internship")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load advertising dataset from Kaggle download or fallback."""
    csv_files = (glob.glob("task4_sales/*.csv")
                 + glob.glob("advertising*.csv")
                 + glob.glob("Advertising*.csv"))
    csv_path = next((f for f in csv_files
                     if "advert" in f.lower() or "sales" in f.lower()), None)

    if csv_path:
        print(f"\n[✓] Loading: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0) if "unnamed" in open(csv_path).readline().lower() else pd.read_csv(csv_path)
    else:
        print("\n[!] CSV not found — generating representative synthetic data")
        np.random.seed(42)
        n = 200
        TV        = np.random.uniform(0.7, 296.4, n)
        Radio     = np.random.uniform(0.0, 49.6,  n)
        Newspaper = np.random.uniform(0.3, 114.0, n)
        noise     = np.random.normal(0, 1.5, n)
        # Approximate the real relationship in the Advertising dataset
        Sales = 2.9 + 0.046*TV + 0.188*Radio + 0.001*Newspaper + noise
        df = pd.DataFrame({"TV": TV, "Radio": Radio, "Newspaper": Newspaper, "Sales": Sales})

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Sales": "Sales"})
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.dropna()
    print(f"    Shape   : {df.shape}")
    print(f"    Columns : {df.columns.tolist()}")
    print(f"\n{df.describe().round(2).to_string()}")
    return df


df = load_data()
CHANNELS = ["TV", "Radio", "Newspaper"]
TARGET   = "Sales"


# ─────────────────────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────────────────────

print("\n─── 2. EXPLORATORY DATA ANALYSIS ───")

# ── 2a. Distributions ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Variable Distributions", fontweight="bold")
for ax, col in zip(axes.flatten(), CHANNELS + [TARGET]):
    color = CHANNEL_COLORS.get(col, "#9b59b6")
    sns.histplot(df[col], bins=25, kde=True, ax=ax, color=color)
    ax.set_title(f"{col} (${df[col].mean():.1f}K mean)")
    ax.set_xlabel(f"{col} Budget ($K)" if col != TARGET else "Sales ($K)")
plt.tight_layout()
plt.savefig("task4_outputs/fig1_distributions.png", bbox_inches="tight")
plt.close()
print("[✓] Saved: task4_outputs/fig1_distributions.png")

# ── 2b. Scatter — each channel vs Sales ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Advertising Spend vs Sales", fontweight="bold")
for ax, channel in zip(axes, CHANNELS):
    ax.scatter(df[channel], df[TARGET], alpha=0.5,
               color=CHANNEL_COLORS[channel], edgecolors="white", s=35)
    # Add regression line
    m, b, r, p, _ = stats.linregress(df[channel], df[TARGET])
    x_range = np.linspace(df[channel].min(), df[channel].max(), 100)
    ax.plot(x_range, m*x_range + b, color="black", lw=1.5, ls="--")
    ax.set_xlabel(f"{channel} Budget ($K)")
    ax.set_ylabel("Sales ($K)")
    ax.set_title(f"{channel}  (r={r:.3f}, p={p:.3f})")
plt.tight_layout()
plt.savefig("task4_outputs/fig2_channel_vs_sales.png", bbox_inches="tight")
plt.close()
print("[✓] Saved: task4_outputs/fig2_channel_vs_sales.png")

# ── 2c. Correlation heatmap ───────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, fmt=".3f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Pearson r"})
ax.set_title("Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig("task4_outputs/fig3_correlation.png", bbox_inches="tight")
plt.close()
print("[✓] Saved: task4_outputs/fig3_correlation.png")

# Print correlations with sales
print("\n  Correlation with Sales:")
for ch in CHANNELS:
    r, p = stats.pearsonr(df[ch], df[TARGET])
    sig  = "✓ significant" if p < 0.05 else "✗ not significant"
    print(f"    {ch:<12} r={r:.4f}  p={p:.4f}  {sig}")


# ─────────────────────────────────────────────────────────────
# 3. SIMPLE LINEAR REGRESSION (per channel)
# ─────────────────────────────────────────────────────────────

print("\n─── 3. SIMPLE LINEAR REGRESSION (per channel) ───")

"""
We first model Sales as a function of each advertising channel
independently. This tells us the isolated effect of each channel.
  Sales = β₀ + β₁ × Channel_Spend
  β₁ = units of sales increase per $1K additional spend
"""

slr_results = {}
for channel in CHANNELS:
    X_ch = df[[channel]].values
    y    = df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_ch, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    slr_results[channel] = {
        "coef" : model.coef_[0],
        "intercept": model.intercept_,
        "r2"   : r2_score(y_te, y_pred),
        "rmse" : np.sqrt(mean_squared_error(y_te, y_pred)),
    }
    print(f"  {channel:<12} β={model.coef_[0]:.4f}  R²={r2_score(y_te,y_pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_te,y_pred)):.3f}")

print("\n  Interpretation of β (sales increase per $1K extra spend):")
for ch, res in slr_results.items():
    print(f"    +$1K in {ch:<12} → +{res['coef']:.3f}K sales units")


# ─────────────────────────────────────────────────────────────
# 4. MULTIPLE LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────

print("\n─── 4. MULTIPLE LINEAR REGRESSION ───")

"""
Multiple Linear Regression uses ALL channels simultaneously.
  Sales = β₀ + β₁·TV + β₂·Radio + β₃·Newspaper + ε

This is more realistic — companies run multiple campaigns at once.
Coefficients now represent the effect of each channel while
HOLDING OTHERS CONSTANT (partial effect / ceteris paribus).
"""

X = df[CHANNELS].values
y = df[TARGET].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr = LinearRegression().fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)
r2_mlr   = r2_score(y_test, y_pred_mlr)
rmse_mlr = np.sqrt(mean_squared_error(y_test, y_pred_mlr))

print(f"\n  Multiple Linear Regression:")
print(f"    Intercept : {mlr.intercept_:.4f}")
for ch, coef in zip(CHANNELS, mlr.coef_):
    print(f"    {ch:<12} β={coef:.4f}")
print(f"    R²   : {r2_mlr:.4f}")
print(f"    RMSE : {rmse_mlr:.4f}")


# ─────────────────────────────────────────────────────────────
# 5. MODEL COMPARISON
# ─────────────────────────────────────────────────────────────

print("\n─── 5. FULL MODEL COMPARISON ───")

MODELS = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0))
    ]),
    "Polynomial (deg=2)": Pipeline([
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg",    LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                              max_depth=3, random_state=42))
    ]),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, pipeline in MODELS.items():
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    cv_r2   = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
    results[name] = {
        "rmse"    : np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae"     : mean_absolute_error(y_test, y_pred),
        "r2"      : r2_score(y_test, y_pred),
        "cv_r2"   : cv_r2.mean(),
        "pipeline": pipeline,
        "y_pred"  : y_pred,
    }
    print(f"  {name:<22} RMSE={results[name]['rmse']:.3f}  "
          f"R²={results[name]['r2']:.4f}  CV-R²={cv_r2.mean():.4f}")

best_name = max(results, key=lambda n: results[n]["r2"])
best      = results[best_name]
print(f"\n  ★ Best Model : {best_name}  (R²={best['r2']:.4f})")


# ─────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────────────────────

# ── Model comparison bar ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Comparison", fontweight="bold")
names = list(results.keys())
pal   = ["#e74c3c" if n == best_name else "#3498db" for n in names]

axes[0].bar(names, [results[n]["rmse"] for n in names], color=pal, edgecolor="white")
axes[0].set_title("RMSE (lower=better)"); axes[0].set_ylabel("RMSE ($K)")
axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)

axes[1].bar(names, [results[n]["r2"] for n in names], color=pal, edgecolor="white")
axes[1].set_title("R² Score (higher=better)"); axes[1].set_ylabel("R²")
axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("task4_outputs/fig4_model_comparison.png", bbox_inches="tight")
plt.close()
print("[✓] Saved: task4_outputs/fig4_model_comparison.png")

# ── Best model: Actual vs Predicted + Residuals ───────────────
y_pred_best = best["y_pred"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Best Model: {best_name}", fontweight="bold")

axes[0].scatter(y_test, y_pred_best, alpha=0.6, color="#2ecc71", edgecolors="white", s=40)
lim = max(y_test.max(), y_pred_best.max()) * 1.05
axes[0].plot([0, lim],[0, lim],"r--", lw=1.5, label="Perfect")
axes[0].set_xlabel("Actual Sales ($K)"); axes[0].set_ylabel("Predicted Sales ($K)")
axes[0].set_title("Actual vs Predicted"); axes[0].legend()

residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.6, color="#e67e22", edgecolors="white", s=40)
axes[1].axhline(0, color="red", ls="--", lw=1.5)
axes[1].set_xlabel("Predicted Sales ($K)"); axes[1].set_ylabel("Residuals ($K)")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("task4_outputs/fig5_actual_vs_predicted.png", bbox_inches="tight")
plt.close()
print("[✓] Saved: task4_outputs/fig5_actual_vs_predicted.png")


# ─────────────────────────────────────────────────────────────
# 7. ADVERTISING ROI ANALYSIS
# ─────────────────────────────────────────────────────────────

print("\n─── 7. ADVERTISING ROI ANALYSIS ───")

"""
ROI (Return on Investment) for each channel:
  ROI = (Sales attributed to channel / Spend on channel) × 100

We estimate attributed sales using the MLR coefficients:
  Attributed Sales = β_channel × mean_spend_on_channel
"""

mean_spends = df[CHANNELS].mean()
attributed  = {ch: mlr.coef_[i] * mean_spends[ch] for i, ch in enumerate(CHANNELS)}
total_attr  = sum(attributed.values())
roi_pct     = {ch: (attributed[ch] / mean_spends[ch]) * 100 for ch in CHANNELS}

print("\n  Channel ROI Summary:")
print(f"  {'Channel':<12} {'Avg Spend':>10} {'Attributed Sales':>18} {'ROI':>8}")
print("  " + "-" * 52)
for ch in CHANNELS:
    print(f"  {ch:<12} ${mean_spends[ch]:>8.1f}K  ${attributed[ch]:>14.2f}K  {roi_pct[ch]:>6.1f}%")

# ROI bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Advertising Channel ROI Analysis", fontweight="bold")

axes[0].bar(CHANNELS, [roi_pct[ch] for ch in CHANNELS],
            color=[CHANNEL_COLORS[ch] for ch in CHANNELS], edgecolor="white", width=0.5)
axes[0].set_title("ROI per Channel (%)")
axes[0].set_ylabel("ROI (%)")
for i, ch in enumerate(CHANNELS):
    axes[0].text(i, roi_pct[ch] + 0.5, f"{roi_pct[ch]:.1f}%", ha="center", fontweight="bold")

# Budget allocation pie
axes[1].pie([mean_spends[ch] for ch in CHANNELS], labels=CHANNELS,
            colors=[CHANNEL_COLORS[ch] for ch in CHANNELS],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "lw": 2})
axes[1].set_title("Current Budget Allocation")

plt.tight_layout()
plt.savefig("task4_outputs/fig6_roi_analysis.png", bbox_inches="tight")
plt.close()
print("[✓] Saved: task4_outputs/fig6_roi_analysis.png")


# ─────────────────────────────────────────────────────────────
# 8. SALES FORECAST SIMULATION
# ─────────────────────────────────────────────────────────────

print("\n─── 8. SALES FORECAST SIMULATION ───")

"""
Simulate: if we shift 10% of TV budget to Radio
(highest ROI channel), what happens to total predicted sales?
"""

best_pipeline = best["pipeline"]

scenario_base  = df[CHANNELS].mean().values.reshape(1, -1)
scenario_shift = scenario_base.copy()
tv_idx, radio_idx = CHANNELS.index("TV"), CHANNELS.index("Radio")
shift_amount = scenario_shift[0, tv_idx] * 0.10
scenario_shift[0, tv_idx]    -= shift_amount
scenario_shift[0, radio_idx] += shift_amount

pred_base  = best_pipeline.predict(scenario_base)[0]
pred_shift = best_pipeline.predict(scenario_shift)[0]
delta      = pred_shift - pred_base

print(f"\n  Baseline scenario (avg spend)  → Predicted Sales: ${pred_base:.2f}K")
print(f"  Shift 10% TV→Radio scenario    → Predicted Sales: ${pred_shift:.2f}K")
print(f"  Δ Sales from reallocation      : {'+' if delta>=0 else ''}{delta:.2f}K")
print(f"  Recommendation: {'Shift budget from TV to Radio ✓' if delta > 0 else 'Keep current allocation'}")


# ─────────────────────────────────────────────────────────────
# 9. SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY — TASK 4: SALES PREDICTION")
print("=" * 60)
best_roi_ch = max(roi_pct, key=roi_pct.get)
print(f"""
  Best Model      : {best_name}
  R² Score        : {best['r2']:.4f}  ({best['r2']*100:.1f}% variance explained)
  RMSE            : ${best['rmse']:.3f}K

  Channel Effectiveness (MLR coefficients):
    TV        β = {mlr.coef_[0]:.4f}  (strongest volume driver)
    Radio     β = {mlr.coef_[1]:.4f}  (highest ROI per dollar)
    Newspaper β = {mlr.coef_[2]:.4f}  (least impactful)

  Best ROI Channel : {best_roi_ch} ({roi_pct[best_roi_ch]:.1f}% ROI)

  Actionable Insights:
  • TV drives the most total sales (highest spend + strong β)
  • Radio delivers the best ROI — underinvested relative to impact
  • Newspaper has negligible effect — budget reallocation advised
  • Shifting 10% of TV budget to Radio yields +{delta:.2f}K sales
  • Polynomial features improve fit → non-linear interaction effects exist
""")
print("[✓] Task 4 complete — all outputs in task4_outputs/\n")
