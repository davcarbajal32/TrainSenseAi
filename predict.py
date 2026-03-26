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
            if feature == "rolling_fatigue" and value > 45:
                explanations.append(f"High recent fatigue ({value:.1f}) suggests recovery is needed")
            elif feature == "noisy_sleep_hours" and value < 6:
                explanations.append(f"Low sleep ({value:.1f} hrs) reduced your readiness")
            elif feature == "sleep_quality" and value < 6:
                explanations.append(f"Poor sleep quality (estimated {value}/10) impacted recovery")
            elif feature == "fatigue" and value > 40:
                explanations.append(f"High workload ({value:.1f}) increased overall fatigue")
            elif feature == "intensity" and value > 5:
                explanations.append(f"High training intensity ({value:.1f}) requires more recovery")

        elif prediction == "Train Hard":
            if feature == "rolling_fatigue" and value < 30:
                explanations.append(f"Low recent fatigue ({value:.1f}) supports harder training")
            elif feature == "noisy_sleep_hours" and value >= 7:
                explanations.append(f"Good sleep ({value:.1f} hrs) supports performance")
            elif feature == "sleep_quality" and value >= 7:
                explanations.append(f"High sleep quality (estimated {value}/10) enhances readiness")
            elif feature == "fatigue" and value < 25:
                explanations.append(f"Low workload ({value:.1f}) means you're fresh for training")
            elif feature == "intensity" and value < 3:
                explanations.append(f"Low recent intensity ({value:.1f}) leaves room to push harder")

        elif prediction == "Light":
            if feature == "rolling_fatigue" and value >= 30 and value <= 45:
                explanations.append(f"Moderate recent fatigue ({value:.1f}) suggests balanced effort")
            elif feature == "noisy_sleep_hours" and value >= 6 and value < 7:
                explanations.append(f"Decent sleep ({value:.1f} hrs) supports moderate training")
            elif feature == "sleep_quality" and value >= 5 and value < 7:
                explanations.append(f"Average sleep quality (estimated {value}/10) supports light work")
            elif feature == "fatigue" and value >= 25 and value <= 40:
                explanations.append(f"Moderate workload ({value:.1f}) suggests keeping it light today")
            elif feature == "intensity" and value >= 3 and value <= 5:
                explanations.append(f"Moderate recent intensity ({value:.1f}) suggests an easier session")

    if not explanations:
        if prediction == "Rest":
            explanations.append("Your combined fatigue and recovery indicators suggest taking a rest day")
        elif prediction == "Train Hard":
            explanations.append("Your recovery and fatigue levels indicate you're ready to push it today")
        elif prediction == "Light":
            explanations.append("Your recovery and fatigue are balanced — a moderate session is ideal today")

    return explanations


def output(input_data):
    model_input = input_data.copy()
    model_input.pop("email", None)

    # Remove fields the model doesn't use
    model_input.pop("very_active_min", None)
    model_input.pop("fairly_active_min", None)
    model_input.pop("lightly_active_min", None)

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