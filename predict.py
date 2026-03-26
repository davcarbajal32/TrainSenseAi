import pickle
import pandas as pd
import requests
import json

# ============================================================
# LOAD MODEL AND METADATA
# ============================================================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("readiness_stats.pkl", "rb") as f:
    readiness_stats = pickle.load(f)

# ============================================================
# YOUR API KEY - replace with your actual key
# ============================================================
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"


def predict_readiness(input_data):
    """
    Use the trained ML model to predict tomorrow's readiness score.
    Returns the predicted score and a context dict with all the numbers.
    """
    model_input = {}
    for col in feature_columns:
        model_input[col] = input_data.get(col, 0)

    df = pd.DataFrame([model_input])
    predicted_readiness = float(model.predict(df)[0])

    # Categorize the score relative to the training distribution
    if predicted_readiness <= readiness_stats["q25"]:
        readiness_level = "low"
    elif predicted_readiness >= readiness_stats["q75"]:
        readiness_level = "high"
    else:
        readiness_level = "moderate"

    context = {
        "predicted_readiness": round(predicted_readiness, 2),
        "readiness_level": readiness_level,
        "readiness_range": {
            "min": readiness_stats["min"],
            "max": readiness_stats["max"],
            "avg": round(readiness_stats["mean"], 2)
        },
        "sleep_hours": input_data.get("sleep_hours", 0),
        "sleep_quality": input_data.get("sleep_quality", 0),
        "fatigue": input_data.get("fatigue", 0),
        "rolling_fatigue_3d": input_data.get("rolling_fatigue_3d", 0),
        "rolling_fatigue_7d": input_data.get("rolling_fatigue_7d", 0),
        "intensity": input_data.get("intensity", 0),
        "recovery_debt": input_data.get("recovery_debt", 0),
        "fatigue_trend": input_data.get("fatigue_trend", 0),
        "heart_rate": input_data.get("heart_rate", 0),
        "TotalSteps": input_data.get("TotalSteps", 0),
        "TotalCalories": input_data.get("TotalCalories", 0)
    }

    return predicted_readiness, context


def get_llm_recommendation(context):
    """
    Send the ML model's numerical output to Claude for a natural language
    workout recommendation with reasoning.
    """
    prompt = f"""You are a sports science advisor inside a fitness app called FitLedger.

A user's data has been analyzed by our ML model. Based on the numbers below, give them a clear workout recommendation for today.

## User's Data (from ML model):
- Predicted Readiness Score: {context['predicted_readiness']} (scale: {context['readiness_range']['min']:.0f} to {context['readiness_range']['max']:.0f}, average is {context['readiness_range']['avg']})
- Readiness Level: {context['readiness_level']}
- Sleep: {context['sleep_hours']} hours (estimated quality: {context['sleep_quality']}/10)
- Current Fatigue: {context['fatigue']:.1f}
- 3-Day Average Fatigue: {context['rolling_fatigue_3d']:.1f}
- 7-Day Average Fatigue: {context['rolling_fatigue_7d']:.1f}
- Fatigue Trend: {context['fatigue_trend']:.1f} (positive = fatigue rising, negative = fatigue dropping)
- Recovery Debt: {context['recovery_debt']:.1f} (negative = behind on sleep)
- Yesterday's Intensity: {context['intensity']:.1f}
- Resting Heart Rate: {context['heart_rate']} bpm
- Steps: {context['TotalSteps']:.0f}
- Calories Burned: {context['TotalCalories']:.0f}

## Instructions:
1. Start with a clear recommendation: "Train Hard", "Light Training", or "Rest Day"
2. Give 2-3 short bullet points explaining WHY based on the numbers
3. Suggest a specific workout type that fits (e.g., "heavy compound lifts", "30-min easy jog", "yoga and stretching")
4. If recommending rest or light, mention when they might be ready to push harder

Keep it conversational and motivating. No more than 150 words total. Do not use markdown headers."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=15
        )

        data = response.json()

        if "content" in data and len(data["content"]) > 0:
            return data["content"][0]["text"]
        else:
            return None

    except Exception as e:
        print(f"LLM API error: {e}")
        return None


def fallback_recommendation(context):
    """
    If the LLM API fails, generate a basic recommendation from the numbers.
    """
    score = context["predicted_readiness"]
    level = context["readiness_level"]

    if level == "high":
        recommendation = "Train Hard"
        explanation = [
            f"Predicted readiness is strong at {score} (above average)",
            f"Sleep of {context['sleep_hours']} hrs is supporting recovery",
            "Your body is primed for a challenging session — heavy lifts or high-intensity intervals"
        ]
    elif level == "low":
        recommendation = "Rest Day"
        explanation = [
            f"Predicted readiness is low at {score} (below average)",
            f"3-day fatigue trend at {context['rolling_fatigue_3d']:.1f} indicates accumulated stress",
            "Focus on recovery — light stretching, hydration, and quality sleep tonight"
        ]
    else:
        recommendation = "Light Training"
        explanation = [
            f"Predicted readiness is moderate at {score}",
            f"Fatigue is manageable but recovery debt is {context['recovery_debt']:.1f}",
            "A moderate session works — easy cardio, mobility work, or light technique practice"
        ]

    return recommendation, explanation


def output(input_data):
    """
    Main entry point called by app.py.
    1. ML model predicts readiness score
    2. LLM generates natural language recommendation
    3. Falls back to rule-based if LLM fails
    """
    predicted_readiness, context = predict_readiness(input_data)

    # Try LLM first
    llm_response = get_llm_recommendation(context)

    if llm_response:
        # Parse the recommendation category from the LLM response
        response_lower = llm_response.lower()
        if "train hard" in response_lower:
            recommendation = "Train Hard"
        elif "rest" in response_lower:
            recommendation = "Rest Day"
        else:
            recommendation = "Light Training"

        # Split LLM response into lines for the explanation list
        lines = [line.strip() for line in llm_response.split("\n") if line.strip()]
        explanation = lines

        return recommendation, explanation, round(predicted_readiness, 2)

    else:
        # Fallback if API fails
        recommendation, explanation = fallback_recommendation(context)
        return recommendation, explanation, round(predicted_readiness, 2)