import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# ============================================================
# LOAD AND MERGE DATA
# ============================================================

act = pd.read_csv("dailyActivity.csv")
act2 = pd.read_csv("dailyActivity(2).csv")

activity = pd.concat([act, act2], ignore_index=True)
activity = activity.drop_duplicates()

sleep = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# ============================================================
# CLEAN AND SELECT FEATURES
# ============================================================

activity = activity[[
    "Id",
    "ActivityDate",
    "TotalSteps",
    "Calories",
    "VeryActiveMinutes",
    "FairlyActiveMinutes",
    "LightlyActiveMinutes",
    "SedentaryMinutes"
]]

sleep = sleep[[
    "Sleep Duration",
    "Quality of Sleep",
    "Heart Rate"
]]

sleep = sleep.rename(columns={
    "Sleep Duration": "sleep_hours",
    "Quality of Sleep": "sleep_quality",
    "Heart Rate": "heart_rate"
})

sleep = sleep.dropna()

activity = activity.rename(columns={
    "Calories": "TotalCalories"
})

activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"])
activity = activity.sort_values(by=["Id", "ActivityDate"])

# Sample sleep data onto activity rows (same as before)
activity["sleep_hours"] = sleep["sleep_hours"].sample(len(activity), replace=True).values
activity["sleep_quality"] = sleep["sleep_quality"].sample(len(activity), replace=True).values
activity["heart_rate"] = sleep["heart_rate"].sample(len(activity), replace=True).values

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Intensity: weighted combination of active minutes
activity["intensity"] = (
    activity["VeryActiveMinutes"] * 3 +
    activity["FairlyActiveMinutes"] * 2 +
    activity["LightlyActiveMinutes"]
) / 60

# Fatigue: combination of calorie burn, steps, and intensity
activity["fatigue"] = (
    activity["TotalCalories"] / 100 +
    activity["TotalSteps"] / 2000 +
    activity["intensity"]
)

# Rolling features (per user, over last 3 days)
activity["rolling_fatigue_3d"] = (
    activity.groupby("Id")["fatigue"]
    .rolling(3).mean()
    .reset_index(level=0, drop=True)
)

activity["rolling_intensity_3d"] = (
    activity.groupby("Id")["intensity"]
    .rolling(3).mean()
    .reset_index(level=0, drop=True)
)

activity["rolling_sleep_3d"] = (
    activity.groupby("Id")["sleep_hours"]
    .rolling(3).mean()
    .reset_index(level=0, drop=True)
)

# 7-day rolling features for longer trends
activity["rolling_fatigue_7d"] = (
    activity.groupby("Id")["fatigue"]
    .rolling(7).mean()
    .reset_index(level=0, drop=True)
)

activity["rolling_sleep_7d"] = (
    activity.groupby("Id")["sleep_hours"]
    .rolling(7).mean()
    .reset_index(level=0, drop=True)
)

# Readiness score (what we want to PREDICT for tomorrow)
activity["readiness"] = (
    activity["sleep_hours"] * activity["sleep_quality"]
) - activity["rolling_fatigue_3d"]

# Recovery debt: difference between 7-day sleep average and ideal (7.5 hrs)
activity["recovery_debt"] = activity["rolling_sleep_7d"] - 7.5

# Fatigue trend: is fatigue going up or down? (3-day vs 7-day)
activity["fatigue_trend"] = activity["rolling_fatigue_3d"] - activity["rolling_fatigue_7d"]

# ============================================================
# CREATE TARGET: next day's readiness (shift up by 1 per user)
# ============================================================

activity["next_day_readiness"] = (
    activity.groupby("Id")["readiness"].shift(-1)
)

# Drop rows with missing values
activity = activity.dropna()

# ============================================================
# MODEL FEATURES AND TARGET
# ============================================================

feature_columns = [
    "sleep_hours",
    "sleep_quality",
    "TotalSteps",
    "TotalCalories",
    "heart_rate",
    "intensity",
    "fatigue",
    "rolling_fatigue_3d",
    "rolling_intensity_3d",
    "rolling_sleep_3d",
    "rolling_fatigue_7d",
    "rolling_sleep_7d",
    "recovery_debt",
    "fatigue_trend"
]

X = activity[feature_columns]
Y = activity["next_day_readiness"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ============================================================
# TRAIN REGRESSION MODEL
# ============================================================

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, Y_train)

# ============================================================
# EVALUATE
# ============================================================

Y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

# ============================================================
# SAVE MODEL AND METADATA
# ============================================================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

# Save readiness stats so we can contextualize scores later
readiness_stats = {
    "mean": float(activity["readiness"].mean()),
    "std": float(activity["readiness"].std()),
    "min": float(activity["readiness"].min()),
    "max": float(activity["readiness"].max()),
    "q25": float(activity["readiness"].quantile(0.25)),
    "q75": float(activity["readiness"].quantile(0.75))
}

with open("readiness_stats.pkl", "wb") as f:
    pickle.dump(readiness_stats, f)

print(f"\nReadiness stats: {readiness_stats}")
print("\nModel, feature columns, and readiness stats saved.")