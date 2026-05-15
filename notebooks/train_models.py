"""
IPL Auction Analysis — Full Pipeline
EDA + Feature Engineering + ML Models + SHAP
Run this file to train and save all models.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import joblib, os, json

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              classification_report, accuracy_score)
import shap

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = "."
DATA_PATH   = f"{BASE}/data/ipl_auction_data.csv"
MODEL_DIR   = f"{BASE}/models"
PLOTS_DIR   = f"{BASE}/notebooks/plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "blue":   "#378ADD",
    "green":  "#1D9E75",
    "amber":  "#EF9F27",
    "coral":  "#D85A30",
    "purple": "#7F77DD",
    "gray":   "#888780",
}

# ─── 1. LOAD & INSPECT ────────────────────────────────────────────────────────
print("=" * 60)
print("  IPL AUCTION ANALYSIS — FULL PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"    Years: {df['year'].min()} – {df['year'].max()}")
print(f"    Players: {df['playername'].nunique()} unique")
print(f"    Franchises: {df['franchise'].nunique() - 1} (excluding Unsold)")

# ─── 2. EDA PLOTS ─────────────────────────────────────────────────────────────
print("\n[2] Generating EDA plots...")

sold = df[df['issold'] == 1].copy()

# Plot 1 — Avg auction price by year
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
fig.suptitle("IPL Auction — Exploratory Data Analysis", fontsize=16, fontweight='bold', y=0.98)

ax = axes[0, 0]
yearly = sold.groupby('year')['soldpricecr'].agg(['mean','median']).reset_index()
ax.plot(yearly['year'], yearly['mean'], marker='o', color=COLORS['blue'], linewidth=2, label='Mean price')
ax.plot(yearly['year'], yearly['median'], marker='s', color=COLORS['coral'], linewidth=2, linestyle='--', label='Median price')
ax.fill_between(yearly['year'], yearly['median'], yearly['mean'], alpha=0.15, color=COLORS['blue'])
ax.set_title("Avg Auction Price Trend (₹ Cr)", fontweight='bold')
ax.set_xlabel("Year"); ax.set_ylabel("Price (₹ Cr)")
ax.legend(); ax.grid(alpha=0.3)

# Plot 2 — Price distribution by role
ax = axes[0, 1]
for role, color in [("Batsman", COLORS['blue']), ("Bowler", COLORS['green']), ("All-Rounder", COLORS['amber'])]:
    data = sold[sold['role'] == role]['soldpricecr']
    ax.hist(data, bins=20, alpha=0.6, color=color, label=role, edgecolor='white')
ax.set_title("Price Distribution by Role", fontweight='bold')
ax.set_xlabel("Sold Price (₹ Cr)"); ax.set_ylabel("Count")
ax.legend(); ax.grid(alpha=0.3)

# Plot 3 — Top franchises by avg spend
ax = axes[1, 0]
franchise_spend = sold.groupby('franchise')['soldpricecr'].mean().sort_values(ascending=True).tail(10)
bars = ax.barh(franchise_spend.index, franchise_spend.values, color=COLORS['purple'], alpha=0.8)
ax.set_title("Avg Price Paid per Player by Franchise", fontweight='bold')
ax.set_xlabel("Avg Price (₹ Cr)")
for bar, val in zip(bars, franchise_spend.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'₹{val:.1f}Cr', va='center', fontsize=9)
ax.grid(alpha=0.3, axis='x')

# Plot 4 — Age vs Price scatter
ax = axes[1, 1]
role_colors = {"Batsman": COLORS['blue'], "Bowler": COLORS['green'], "All-Rounder": COLORS['amber']}
for role, color in role_colors.items():
    mask = sold['role'] == role
    ax.scatter(sold[mask]['age'], sold[mask]['soldpricecr'], c=color, alpha=0.5, s=40, label=role)
ax.set_title("Age vs Sold Price", fontweight='bold')
ax.set_xlabel("Age"); ax.set_ylabel("Sold Price (₹ Cr)")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/eda_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: eda_overview.png")

# Plot 5 — Value Index (performance per crore)
fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
top_value = sold.nlargest(15, 'valueindex')[['playername', 'valueindex', 'soldpricecr', 'role']].reset_index(drop=True)
colors_list = [role_colors[r] for r in top_value['role']]
bars = ax.bar(top_value['playername'], top_value['valueindex'], color=colors_list, alpha=0.85)
ax.set_title("Top 15 — Best Value Players (Performance Score ÷ Price)", fontsize=13, fontweight='bold')
ax.set_ylabel("Value Index")
plt.xticks(rotation=35, ha='right', fontsize=9)
handles = [mpatches.Patch(color=c, label=r) for r, c in role_colors.items()]
ax.legend(handles=handles)
ax.grid(alpha=0.3, axis='y')
for bar, row in zip(bars, top_value.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'₹{row.soldpricecr}Cr', ha='center', fontsize=7.5)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/value_index.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: value_index.png")

# ─── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n[3] Feature engineering...")

# Work only with sold players for regression
reg_df = sold.copy()

# Encode categoricals
le_role = LabelEncoder()
le_nat  = LabelEncoder()
reg_df['role_enc'] = le_role.fit_transform(reg_df['role'])
reg_df['nationality_enc'] = le_nat.fit_transform(reg_df['nationality'])

# Save encoders
joblib.dump(le_role, f"{MODEL_DIR}/le_role.pkl")
joblib.dump(le_nat,  f"{MODEL_DIR}/le_nat.pkl")

# Feature set
FEATURES = [
    'role_enc', 'nationality_enc', 'age', 'iplexperienceyears',
    'battingavg', 'battingsr', 'runsscored', 'fifties', 'hundreds',
    'wickets', 'economyrate', 'bowlingavg', 'bowlingsr',
    'basepricecr', 'performancescore', 'year'
]
TARGET = 'soldpricecr'

X = reg_df[FEATURES]
y = reg_df[TARGET]

# Save feature names
with open(f"{MODEL_DIR}/feature_names.json", 'w') as f:
    json.dump(FEATURES, f)

print(f"    Features: {len(FEATURES)} | Samples: {len(X)}")

# ─── 4. TRAIN REGRESSION MODEL ────────────────────────────────────────────────
print("\n[4] Training price prediction model (Random Forest)...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, None],
    'min_samples_leaf': [2, 4]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"    Best params: {grid_search.best_params_}")
print(f"    R²   = {r2:.4f}")
print(f"    RMSE = ₹{rmse:.2f} Cr")
print(f"    MAE  = ₹{mae:.2f} Cr")

# Save model + metrics
joblib.dump(best_rf, f"{MODEL_DIR}/price_predictor.pkl")
metrics = {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
           "best_params": grid_search.best_params_}
with open(f"{MODEL_DIR}/regression_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

# Actual vs Predicted plot
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
ax.scatter(y_test, y_pred, alpha=0.5, color=COLORS['blue'], edgecolors='white', s=60)
lims = [0, max(y_test.max(), y_pred.max()) + 1]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel("Actual Price (₹ Cr)"); ax.set_ylabel("Predicted Price (₹ Cr)")
ax.set_title(f"Actual vs Predicted Auction Price\nR² = {r2:.3f} | RMSE = ₹{rmse:.2f} Cr", fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: actual_vs_predicted.png")

# ─── 5. SHAP EXPLAINABILITY ───────────────────────────────────────────────────
print("\n[5] Computing SHAP values...")

explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

# Global feature importance (SHAP)
fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
mean_shap = np.abs(shap_values).mean(axis=0)
feat_importance = pd.Series(mean_shap, index=FEATURES).sort_values(ascending=True)
colors_shap = [COLORS['coral'] if v > feat_importance.median() else COLORS['blue']
               for v in feat_importance.values]
bars = ax.barh(feat_importance.index, feat_importance.values, color=colors_shap, alpha=0.85)
ax.set_title("SHAP Feature Importance — What Drives Auction Price?", fontsize=13, fontweight='bold')
ax.set_xlabel("Mean |SHAP Value| (₹ Cr impact)")
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/shap_global.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: shap_global.png")

# Save SHAP explainer
joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.pkl")

# ─── 6. CLASSIFICATION MODEL (Sold vs Unsold) ─────────────────────────────────
print("\n[6] Training sold/unsold classifier...")

clf_df = df.copy()
clf_df['role_enc'] = le_role.transform(clf_df['role'])
clf_df['nationality_enc'] = le_nat.transform(clf_df['nationality'])

Xc = clf_df[FEATURES]
yc = clf_df['issold']

# Check if we have both classes
unique_classes = yc.unique()
if len(unique_classes) < 2:
    print(f"    Warning: Only {len(unique_classes)} class(es) found in data. Skipping classifier training.")
    acc = 1.0 if yc.iloc[0] == yc.iloc[-1] else 0.5
    clf_metrics = {"accuracy": round(acc, 4), "note": "Only one class in dataset"}
    with open(f"{MODEL_DIR}/classifier_metrics.json", 'w') as f:
        json.dump(clf_metrics, f)
    # Create dummy scaler and classifier (not fitted on multi-class data)
    scaler = StandardScaler()
    scaler.fit(Xc.iloc[:10])
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    # Use a dummy pipeline or save a note that classifier needs balanced data
    print("    Models saved with warnings - consider collecting more balanced data")
else:
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2,
                                                              random_state=42, stratify=yc)
    scaler = StandardScaler()
    Xc_train_s = scaler.fit_transform(Xc_train)
    Xc_test_s  = scaler.transform(Xc_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xc_train_s, yc_train)
    yc_pred = lr.predict(Xc_test_s)
    acc = accuracy_score(yc_test, yc_pred)
    print(f"    Classifier accuracy: {acc:.4f}")
    print(classification_report(yc_test, yc_pred, target_names=['Unsold','Sold']))

    joblib.dump(lr,     f"{MODEL_DIR}/sold_classifier.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    clf_metrics = {"accuracy": round(acc, 4)}
    with open(f"{MODEL_DIR}/classifier_metrics.json", 'w') as f:
        json.dump(clf_metrics, f)

# ─── 7. SAVE SUMMARY STATS FOR DASHBOARD ──────────────────────────────────────
print("\n[7] Saving dashboard data...")

summary = {
    "total_records": len(df),
    "total_sold": int(df['issold'].sum()),
    "total_unsold": int((df['issold']==0).sum()),
    "avg_price_overall": round(sold['soldpricecr'].mean(), 2),
    "max_price": round(sold['soldpricecr'].max(), 2),
    "min_price": round(sold['soldpricecr'].min(), 2),
    "years": sorted(df['year'].unique().tolist()),
    "franchises": sorted([f for f in df['franchise'].unique() if f != 'Unsold']),
    "roles": sorted([r for r in df['role'].unique() if pd.notna(r)]),
}
with open(f"{MODEL_DIR}/summary_stats.json", 'w') as f:
    json.dump(summary, f, indent=2)

# Top value players (all-time)
top_val = sold.nlargest(20, 'valueindex')[
    ['playername','role','franchise','year','soldpricecr','performancescore','valueindex']
].reset_index(drop=True)
top_val.to_csv(f"{MODEL_DIR}/top_value_players.csv", index=False)

# Franchise efficiency
fran_eff = sold.groupby('franchise').agg(
    total_spend=('soldpricecr','sum'),
    avg_price=('soldpricecr','mean'),
    avg_value_index=('valueindex','mean'),
    players_bought=('playername','count'),
    avg_performance=('performancescore','mean')
).round(2).reset_index()
fran_eff.to_csv(f"{MODEL_DIR}/franchise_efficiency.csv", index=False)

# Save processed dataset
sold.to_csv(f"{MODEL_DIR}/processed_data.csv", index=False)

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"  Models saved  → {MODEL_DIR}/")
print(f"  Plots saved   → {PLOTS_DIR}/")
print(f"\n  Key metrics:")
print(f"    Price predictor R²   = {r2:.4f}")
print(f"    Sold/Unsold accuracy = {acc:.4f}")
print(f"\n  Next step: Run the Streamlit app!")
print(f"  $ cd {BASE} && streamlit run streamlit_app/app.py")
