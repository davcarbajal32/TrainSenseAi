import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_importance.pkl", "rb") as f:
    feature_importance = pickle.load(f)


def generate_explanation(input_data, feature_importance, prediction):
    explanations = []

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    for feature, _ in sorted_features[:3]:
        value = input_data.get(feature)
        if value is None:
            continue

        if prediction == "Rest":
            if feature == "rolling_fatigue" and value > 50:
                explanations.append(f"High recent fatigue ({value:.1f}) suggests recovery")
            elif feature == "noisy_sleep_hours" and value < 6.5:
                explanations.append(f"Low sleep ({value:.1f} hrs) reduced readiness")

        elif prediction == "Train Hard":
            if feature == "rolling_fatigue" and value < 40:
                explanations.append(f"Low fatigue ({value:.1f}) supports training")
            elif feature == "noisy_sleep_hours" and value >= 7:
                explanations.append(f"Good sleep ({value:.1f} hrs) supports performance")

        elif prediction == "Light":
            explanations.append("Moderate fatigue suggests balanced training")

    if not explanations:
        explanations.append("Balanced fatigue and recovery levels")

    return explanations


def output(input_data):
    model_input = input_data.copy()
    model_input.pop("email", None)

    expected_features = [
        "noisy_sleep_hours",
        "sleep_quality",
        "TotalSteps",
        "TotalCalories",
        "heart_rate",
        "intensity",
        "fatigue",
        "rolling_fatigue"
    ]

    df = pd.DataFrame([model_input])

    for col in expected_features:
        if col not in df:
            df[col] = 0

    df = df[expected_features]

    prediction = model.predict(df)[0]
    explanation = generate_explanation(input_data, feature_importance, prediction)

    return prediction, explanation