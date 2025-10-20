import pandas as pd
import pandas as pd
import numpy as np
import sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#most artist bias is removed because we removed artist followers and artist popularity
file_path = "/Users/tasi/Desktop/Capstone Project /data/clean_spotify_post2012.csv"
df = pd.read_csv(file_path)


# R² = 0.3416
# Explains 34.16% of the variation in track popularity. Higher is better.
# Much higher than Linear Regression’s ~0.15.
# Shows that Random Forest captures nonlinear patterns between features and popularity (like combinations of danceability + energy).

# RMSE = 16.23
# Average prediction error (in popularity points). Lower is better.
# Smaller than Decision Tree (≈ 23) and Linear Regression (≈ 18).
# Our predictions are closer to real popularity than before.

# MAE = 12.40
# Average absolute difference between prediction and actual. Lower is better.
# Similar trend — lower = more accurate.
# Confirms consistent improvement.

# ============================================================
#  FEATURE ENGINEERING + MULTICOLLINEARITY CHECK SCRIPT
#  Author: Tasianna Giordano
#  Dataset: spotify_post2012 (DataFrame: df)
# ============================================================

# --- IMPORT REQUIRED LIBRARIES ---
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ============================================================
# 1. CREATE INTERACTION FEATURES
# ============================================================
# Interaction features capture relationships between variables that, when combined,
# may explain popularity better than any single feature.

df['energy_dance'] = df['energy'] * df['danceability']
df['acoustic_instr'] = df['acousticness'] * df['instrumentalness']
df['loud_valence'] = df['loudness'] * df['valence']

# ============================================================
# 2. CREATE POLYNOMIAL (SQUARED) FEATURES
# ============================================================
# Squared terms help capture nonlinear relationships (for example,
# very high or low energy might relate differently to popularity).

df['energy_squared'] = df['energy'] ** 2
df['danceability_squared'] = df['danceability'] ** 2
df['valence_squared'] = df['valence'] ** 2

# ============================================================
# 3. TEMPORAL FEATURE ENGINEERING
# ============================================================
# Derive 'years_since_release' to represent how old the track is.
# This can help reveal whether recency impacts popularity.
# NOTE: 2023 used as dataset endpoint; adjust if dataset updated.

df['years_since_release'] = 2023 - df['release_year']

# ============================================================
# 4. OPTIONAL BINNING (COMMENT OUT IF NOT NEEDED)
# ============================================================
# Categorize continuous variables like tempo and loudness.
# These bins can help classification models or visualization.
# They will be converted to dummy variables later if used.

df['tempo_bin'] = pd.qcut(df['tempo'], q=4, labels=['slow', 'moderate', 'fast', 'very_fast'])
df['loudness_bin'] = pd.cut(df['loudness'], bins=[-60, -20, -10, 0],
                            labels=['low', 'medium', 'high'])

# ============================================================
# 5. REMOVE FEATURES THAT CAUSE PERFECT OR STRONG MULTICOLLINEARITY
# ============================================================
# - 'release_year' and 'years_since_release' are perfectly correlated.
# - Squared terms are highly correlated with their original features.
# We drop some to prevent inflated VIF and unstable model weights.

cols_to_drop = [
    'release_year',          # redundant with years_since_release
    'energy_squared',        # duplicates energy
    'danceability_squared',  # duplicates danceability
    'valence_squared'        # duplicates valence
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# ============================================================
# 6. OPTIONAL: DROP PARENT OR CHILD FEATURE TO REDUCE OVERLAP
# ============================================================
# 'energy_dance' combines two strong features, so we can drop one of them
# to further reduce multicollinearity. Adjust as needed.
df.drop(columns=['energy'], inplace=True, errors='ignore')

# ============================================================
# 7. SELECT ONLY NUMERIC COLUMNS FOR VIF CHECK
# ============================================================
# Non-numeric columns (like tempo_bin or loudness_bin) cannot be used
# in VIF calculations and will cause TypeErrors if included.

X = df.select_dtypes(include=['float64', 'int64'])

# Add a constant term for statsmodels
X_const = add_constant(X)

# ============================================================
# 8. CALCULATE VARIANCE INFLATION FACTOR (VIF)
# ============================================================
# VIF quantifies how much a feature is linearly dependent on others.
# VIF > 10 means high multicollinearity, VIF > 5 is moderate.

vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(X_const.values, i + 1)  # +1 skips the constant column
    for i in range(len(X.columns))
]

print("\n=== VARIANCE INFLATION FACTOR (VIF) RESULTS ===")
print(vif.sort_values("VIF", ascending=False))

# ============================================================
# 9. (OPTIONAL) SAVE THE CLEANED AND ENGINEERED DATASET
# ============================================================
# This lets you use the new features directly for modeling later.

output_path = "../data/spotify_post2012_engineered.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned and engineered dataset saved to: {output_path}")

# ============================================================
# 10. NEXT STEPS:
# ============================================================
# - Re-run your regression/classification models on this engineered dataset.
# - Compare performance (R², RMSE, accuracy).
# - Generate new feature importances to see which engineered features
#   improved predictive power.

# === Define target and feature columns ===
target = 'track_popularity'

# Use only these features (exclude bin features unless doing classification)
feature_cols = [
    'danceability', 'musical_key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature',
    'energy_dance', 'acoustic_instr', 'loud_valence',
    'years_since_release'
]

# === Prepare the feature matrix (X) and target vector (y) ===
X = df[feature_cols]
y = df[target]

# === Split into training and test sets (80/20) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Initialize and fit the Random Forest model ===
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# === Predict on test set ===
y_pred = rf_model.predict(X_test)

# === Evaluate model performance ===
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("=== RANDOM FOREST PERFORMANCE ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# === Feature Importance Visualization ===
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

import matplotlib.pyplot as plt
import numpy as np

# === 1. FEATURE IMPORTANCE ===
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features_sorted = np.array(feature_cols)[indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)", fontsize=14)
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), features_sorted, rotation=45, ha='right')
plt.ylabel("Importance")
plt.tight_layout()
plt.grid(True)
plt.show()


# === 2. PREDICTED VS ACTUAL PLOT ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Track Popularity")
plt.grid(True)
plt.tight_layout()
plt.show()


# === 3. RESIDUALS PLOT ===
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, edgecolors="k")
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Popularity")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot")
plt.grid(True)
plt.tight_layout()
plt.show()
