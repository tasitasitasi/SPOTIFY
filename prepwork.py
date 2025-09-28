
# loading libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# loading dataset
file_path = "../data/clean_spotify_post2012.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print(df.head())

# define target and features
# y is the thing we are predicting
# then creating a list of feature columns and assign to x to use as predictors

y = df['track_popularity']  # target variable - what we are predicting

feature_cols = [
    'danceability', 'energy', 'musical_key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature'
]

X = df[feature_cols]  # loading features from the column above

# split into test and train. 80% of the data is used for training and 20% is used to eval -
# model performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# linear regression model - fits a straight line between features and popularity
# Assumes each feature contributes additively to popularity.
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))  # manual sqrt

print("\n=== Linear Regression ===")
print("R²:", round(r2_lr, 4))
print("RMSE:", round(rmse_lr, 4))

# decision tree model - non linear relationship
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

r2_dt = r2_score(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))  # manual sqrt

print("\n=== Decision Tree ===")
print("R²:", round(r2_dt, 4))
print("RMSE:", round(rmse_dt, 4))


######### RESULTS


# Dataset shape: (62015, 14)
#    track_popularity  danceability  ...  time_signature  release_year
# 0                26         0.319  ...               3        2018.0
# 1                25         0.269  ...               3        2018.0
# 2                18         0.644  ...               3        2020.0
# 3                18         0.627  ...               4        2020.0
# 4                17         0.442  ...               4        2020.0
#
# [5 rows x 14 columns]
#
# === Linear Regression ===
# R²: 0.1556
# RMSE: 18.3773
#
# R² = 0.1556 - Only ~15.6% of the variation in track popularity is explained by the audio features.

# RMSE = 18.38 - On average, predictions are off by ~18 popularity points.

# === Decision Tree ===
# R²: -0.3099
# RMSE: 22.8885

# R² = -0.3099 → Worse than simply predicting the mean popularity for all songs.
#
# RMSE = 22.89 → Larger error than linear regression.


#### VISUALIZE THESE RESULTS

import matplotlib.pyplot as plt

# Linear Regression
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lr, alpha=0.3)
plt.plot([0, 100], [0, 100], 'r--')  # reference line
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Linear Regression: Predicted vs Actual")
plt.show()

# Decision Tree
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_dt, alpha=0.3, color="orange")
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Decision Tree: Predicted vs Actual")
plt.show()


import matplotlib.pyplot as plt

# === Feature Importance (Decision Tree) ===
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]  # sort from most to least important

plt.figure(figsize=(8,6))
plt.bar(range(len(feature_cols)), importances[indices], align="center")
plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Decision Tree Feature Importances")
plt.tight_layout()
plt.show()
