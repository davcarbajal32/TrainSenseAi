import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

act = pd.read_csv("dailyActivity.csv")
act2 = pd.read_csv("dailyActivity(2).csv")

#extending data by combining multiple time windows to improve model generalization
activity = pd.concat([act,act2], ignore_index=True)
activity = activity.drop_duplicates()
sleep = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

#reducing noise in the datasets
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

#organizing dataset
activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"])
activity = activity.sort_values(by=["Id", "ActivityDate"])


activity["sleep_hours"] = sleep["sleep_hours"].sample(len(activity), replace= True).values
activity["sleep_quality"] = sleep["sleep_quality"].sample(len(activity), replace= True).values
activity["heart_rate"] = sleep["heart_rate"].sample(len(activity), replace= True).values

#builing intensity and fatigue features
activity["intensity"] = (activity["VeryActiveMinutes"] * 3 + activity["FairlyActiveMinutes"] * 2 + activity["LightlyActiveMinutes"]) / 60
activity["fatigue"] = (activity["TotalCalories"] / 100 + activity["TotalSteps"] / 2000 + activity["intensity"])

#adding rolling fatigue
activity["rolling_fatigue"] = activity.groupby("Id")["fatigue"].rolling(3).mean().reset_index(level=0, drop= True)

#adding readiness score
activity["readiness"] = (
    activity["sleep_hours"] * activity["sleep_quality"]
) - activity["rolling_fatigue"]

#removing instances with missing data
activity = activity.dropna()

#labeling
def label(row):
    if row["sleep_hours"] < 6.5:
        return "Rest"
    elif row["rolling_fatigue"] > 50:
        return "Rest"
    elif row["sleep_quality"] > 7 and row["rolling_fatigue"] < 35:
        return "Train Hard"
    else:
        return "Light"

activity["label"] = activity.apply(label, axis=1)

#print(activity["label"].value_counts())
#print(activity["label"].value_counts(normalize=True))

#To prevent the model from directly replicating rule-based thresholds, introduced noise to sleep features
#LOOKING FOR OTHER SOLUTIONS
activity["noisy_sleep_hours"] = activity["sleep_hours"] + np.random.normal(0, 0.5, len(activity))

#model building
X = activity[[
    #"sleep_hours",
    "noisy_sleep_hours",
    "sleep_quality",
    "TotalSteps",
    "TotalCalories",
    "heart_rate",
    "intensity",
    "fatigue",
    "rolling_fatigue"
]]

Y = activity["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

#model testing
Y_pred = model.predict(X_test)

#print(classification_report(Y_test, Y_pred))
#print(confusion_matrix(Y_test, Y_pred))

#feature importance
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)
feature_importance = dict(zip(importance_df["feature"], importance_df["importance"]))
#print(importance_df)

def generate_explanation(input_data, feature_importance, prediction):
    explanations = []

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_features[:3]:
        value = input_data[feature]

        if prediction == "Rest":
            if feature == "rolling_fatigue" and value > 50:
                explanations.append(f"High recent fatigue ({value:.1f}) suggests need for recovery")

            elif feature == "noisy_sleep_hours" and value < 6.5:
                explanations.append(f"Lower sleep duration ({value:.1f} hrs) reduced readiness")

            elif feature == "sleep_quality" and value < 6:
                explanations.append(f"Low sleep quality ({value}) impacted recovery")

            elif feature == "fatigue" and value > 40:
                explanations.append(f"High workload ({value:.1f}) increased fatigue")

        elif prediction == "Train Hard":
            if feature == "rolling_fatigue" and value < 40:
                explanations.append(f"Low recent fatigue ({value:.1f}) supports higher training capacity")

            elif feature == "noisy_sleep_hours" and value >= 7:
                explanations.append(f"Sufficient sleep duration ({value:.1f} hrs) supports recovery")

            elif feature == "sleep_quality" and value >= 7:
                explanations.append(f"Good sleep quality ({value}) enhances readiness")

            elif feature == "fatigue" and value < 30:
                explanations.append(f"Low workload ({value:.1f}) indicates readiness for training")

        elif prediction == "Light":
            if feature == "rolling_fatigue":
                explanations.append(f"Moderate fatigue ({value:.1f}) suggests balanced training")

            elif feature == "sleep_quality":
                explanations.append(f"Sleep quality ({value}) supports moderate effort")

    if not explanations:
        explanations.append("Balanced fatigue and recovery levels support this recommendation")

    return explanations

def output(input_data):
    features = pd.DataFrame([input_data])

    prediction = model.predict(features)[0]
    explanation = generate_explanation(input_data, feature_importance, prediction)
    return prediction, explanation

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_importance.pkl", "wb") as f:
    pickle.dump(feature_importance, f)