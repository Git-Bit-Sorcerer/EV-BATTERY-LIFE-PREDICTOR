# EV Battery Health Prediction
# Machine Learning College Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    VotingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# 1. Load dataset
# ----------------------------
print("Loading dataset...")
df = pd.read_csv("ev_battery_health_data.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# ----------------------------
# 2. Basic EDA
# ----------------------------
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between EV Battery Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# ----------------------------
# 3. Prepare data
# ----------------------------
target = "battery_health_%"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ----------------------------
# 4. Train models
# ----------------------------
# ----------------------------------------------------
# Definition of Two-Stage Residual Hybrid Regressor
# ----------------------------------------------------
class TwoStageResidualHybrid(BaseEstimator, RegressorMixin):
    """
    Hybrid Two-Stage Residual Regressor:
    - Stage 1: Fits Linear Regression to capture the overarching linear degradation baseline.
    - Stage 2: Fits Gradient Boosting on residual errors to capture non-linear thermal & cycling dynamics.
    - Output: Linear baseline prediction + Gradient Boosted residual correction.
    """
    def __init__(self, n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.stage1_linear = LinearRegression()
        self.stage2_booster = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.stage1_linear.fit(X, y)
        linear_preds = self.stage1_linear.predict(X)
        residuals = y - linear_preds
        self.stage2_booster.fit(X, residuals)
        return self

    def predict(self, X):
        return self.stage1_linear.predict(X) + self.stage2_booster.predict(X)


# Base models for individual evaluation
base_lr = LinearRegression()
base_dt = DecisionTreeRegressor(
    max_depth=7,
    min_samples_leaf=4,
    random_state=42
)
base_rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
base_gb = GradientBoostingRegressor(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# ----------------------------------------------------
# 3 Distinct Hybrid Models
# ----------------------------------------------------
# Hybrid Model 1: Stacking Regressor
# Combines Linear Regression, Random Forest, and Gradient Boosting with a Ridge meta-regressor.
hybrid_stacking = StackingRegressor(
    estimators=[
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42))
    ],
    final_estimator=Ridge(),
    n_jobs=-1
)

# Hybrid Model 2: Voting / Blending Regressor
# Combines predictions across multiple distinct paradigms via ensemble averaging for variance reduction.
hybrid_voting = VotingRegressor(
    estimators=[
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42))
    ],
    n_jobs=-1
)

# Hybrid Model 3: Two-Stage Residual Regressor
# Additive hybrid combining linear global degradation physics with non-linear boosting on residuals.
hybrid_residual = TwoStageResidualHybrid(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

models = {
    "Linear Regression": base_lr,
    "Decision Tree": base_dt,
    "Random Forest": base_rf,
    "Gradient Boosting": base_gb,
    "Hybrid 1: Stacking Regressor": hybrid_stacking,
    "Hybrid 2: Voting Regressor": hybrid_voting,
    "Hybrid 3: Two-Stage Residual": hybrid_residual
}

results = []
predictions = {}
trained_models = {}

print("\nTraining models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    predictions[name] = pred
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values(
    by="R2 Score", ascending=False
).reset_index(drop=True)

print("Model Results")
print("-" * 65)
print(results_df.to_string(index=False, formatters={
    "MAE": "{:.3f}".format,
    "RMSE": "{:.3f}".format,
    "R2 Score": "{:.4f}".format
}))

# ----------------------------
# 5. Best model
# ----------------------------
best_model_name = results_df.loc[0, "Model"]
best_r2 = results_df.loc[0, "R2 Score"]
best_pred = predictions[best_model_name]

print("\nBest Model:", best_model_name)
print("R2 Score  :", round(best_r2, 4))

# A value around 0.90 or above indicates a strong model.
# We still print the real calculated score instead of fixing it manually.
if best_r2 >= 0.90:
    print("The model explains more than 90% of the variation in battery health.")
elif best_r2 >= 0.80:
    print("The model gives a reasonably good prediction result.")
else:
    print("The model can be improved with better data/features or tuning.")

# ----------------------------
# 6. Compare models
# ----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="R2 Score", y="Model")
plt.xlim(0, 1.05)
plt.title("R2 Score Comparison")
plt.tight_layout()
plt.savefig("model_r2_comparison.png")
plt.close()

# ----------------------------
# 7. Actual vs predicted
# ----------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_pred, alpha=0.6)

minimum = min(y_test.min(), best_pred.min())
maximum = max(y_test.max(), best_pred.max())
plt.plot([minimum, maximum], [minimum, maximum], "--")

plt.xlabel("Actual Battery Health (%)")
plt.ylabel("Predicted Battery Health (%)")
plt.title("Actual vs Predicted - " + best_model_name)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

# ----------------------------
# 8. Feature importance
# ----------------------------
# Random Forest gives feature importance directly.
rf_model = trained_models["Random Forest"]
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nRandom Forest Feature Importance:")
print(importance_df.to_string(index=False, formatters={
    "Importance": "{:.4f}".format
}))

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ----------------------------
# 9. Test with one sample
# ----------------------------
sample = pd.DataFrame([{
    "total_distance_km": 18500.0,
    "average_trip_speed_kmph": 38.5,
    "ambient_temperature_C": 28.0,
    "trip_duration_min": 45.0,
    "charging_cycles": 320,
    "fast_charging_ratio_%": 25.5,
    "average_battery_temperature_C": 35.0
}])

sample_prediction = trained_models[best_model_name].predict(sample)[0]

print("\nSample EV predicted battery health:", round(sample_prediction, 2), "%")
print("\nProject completed successfully.")
