from flask import Flask, request, jsonify, render_template
from predict import output
from database import users, predictions, daily_inputs
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("first.html")


@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    if users.find_one({"email": data["email"]}):
        return jsonify({"message": "Account already exists"})
    users.insert_one(data)
    return jsonify({"message": "User created"})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = users.find_one(data)

    if user:
        return jsonify({"message": "Login successful"})
    return jsonify({"message": "Invalid credentials"})


@app.route("/check_yesterday", methods=["POST"])
def check_yesterday():
    data = request.json
    email = data["email"]

    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Always query the database fresh - no caching
    yesterday_entry = daily_inputs.find_one({
        "email": email,
        "date": yesterday
    })

    today_prediction = predictions.find_one({
        "email": email,
        "date": today
    }, {"_id": 0})

    return jsonify({
        "needs_input": yesterday_entry is None,
        "has_today_prediction": today_prediction is not None,
        "today_prediction": today_prediction if today_prediction else None
    })


def estimate_sleep_quality(sleep_hours):
    """
    Estimate sleep quality (1-10) from sleep duration.
    Mirrors the Sleep_health_and_lifestyle_dataset.csv distribution
    where Sleep Duration and Quality of Sleep are correlated:
      ~5 hrs -> quality 3-4
      ~6 hrs -> quality 5-6
      ~7 hrs -> quality 7
      ~7.5-8 hrs -> quality 8-9
      ~8.5+ hrs -> quality 9-10
    """
    if sleep_hours <= 4:
        return 2
    elif sleep_hours <= 5:
        return 4
    elif sleep_hours <= 5.5:
        return 5
    elif sleep_hours <= 6:
        return 6
    elif sleep_hours <= 6.5:
        return 6
    elif sleep_hours <= 7:
        return 7
    elif sleep_hours <= 7.5:
        return 8
    elif sleep_hours <= 8:
        return 8
    elif sleep_hours <= 8.5:
        return 9
    else:
        return 9


def compute_intensity(very_active, fairly_active, lightly_active):
    """Same formula as training script."""
    return (very_active * 3 + fairly_active * 2 + lightly_active) / 60


def compute_fatigue(calories, steps, intensity):
    """Same formula as training script."""
    return (calories / 100) + (steps / 2000) + intensity


def compute_rolling_fatigue(email, today_fatigue):
    """Average fatigue over last 3 entries including today."""
    past = list(daily_inputs.find(
        {"email": email}
    ).sort("date", -1).limit(2))

    vals = [d["input"]["fatigue"] for d in past]
    vals.append(today_fatigue)

    return sum(vals) / len(vals)


@app.route("/seed")
def seed():
    # Clear old seed data
    users.delete_many({"email": "test"})
    daily_inputs.delete_many({"email": "test"})
    predictions.delete_many({"email": "test"})

    users.insert_one({"email": "test", "password": "test"})

    for i in range(7):
        sleep_hrs = 5 + i * 0.3
        sleep_qual = estimate_sleep_quality(sleep_hrs)
        steps = 4000 + i * 1000
        calories = 200 + i * 50
        very_active = 10 + i * 5
        fairly_active = 15 + i * 3
        lightly_active = 30 + i * 2

        intensity = compute_intensity(very_active, fairly_active, lightly_active)
        fatigue = compute_fatigue(calories, steps, intensity)

        daily_inputs.insert_one({
            "email": "test",
            "date": f"2026-03-{17 + i:02d}",
            "input": {
                "noisy_sleep_hours": sleep_hrs,
                "sleep_quality": sleep_qual,
                "TotalSteps": steps,
                "TotalCalories": calories,
                "heart_rate": 70,
                "very_active_min": very_active,
                "fairly_active_min": fairly_active,
                "lightly_active_min": lightly_active,
                "intensity": intensity,
                "fatigue": fatigue
            }
        })

    return "Seeded"


@app.route("/auto_predict", methods=["POST"])
def auto_predict():
    data = request.json
    email = data["email"]
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    yesterday_entry = daily_inputs.find_one({"email": email, "date": yesterday})

    if not yesterday_entry:
        return jsonify({"error": "No data for yesterday"})

    input_data = yesterday_entry["input"].copy()
    input_data["email"] = email

    rolling = compute_rolling_fatigue(email, input_data.get("fatigue", 0))
    input_data["rolling_fatigue"] = rolling

    prediction, explanation = output(input_data)

    # Only store if we don't already have today's prediction
    if not predictions.find_one({"email": email, "date": today}):
        predictions.insert_one({
            "email": email,
            "date": today,
            "prediction": prediction,
            "explanation": explanation
        })

    return jsonify({
        "prediction": prediction,
        "explanation": explanation
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    email = data["email"]
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Pull raw user inputs
    sleep_hours = float(data.get("noisy_sleep_hours", 0))
    steps = float(data.get("TotalSteps", 0))
    calories = float(data.get("TotalCalories", 0))
    heart_rate = float(data.get("heart_rate", 0))
    very_active = float(data.get("very_active_min", 0))
    fairly_active = float(data.get("fairly_active_min", 0))
    lightly_active = float(data.get("lightly_active_min", 0))

    # Compute derived features server-side (matches training formulas)
    sleep_quality = estimate_sleep_quality(sleep_hours)
    intensity = compute_intensity(very_active, fairly_active, lightly_active)
    fatigue = compute_fatigue(calories, steps, intensity)
    rolling = compute_rolling_fatigue(email, fatigue)

    # Build the stored input
    stored_input = {
        "noisy_sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "TotalSteps": steps,
        "TotalCalories": calories,
        "heart_rate": heart_rate,
        "very_active_min": very_active,
        "fairly_active_min": fairly_active,
        "lightly_active_min": lightly_active,
        "intensity": intensity,
        "fatigue": fatigue
    }

    # Check if yesterday's entry already exists - don't duplicate
    existing = daily_inputs.find_one({"email": email, "date": yesterday})
    if existing:
        # Update instead of duplicate
        daily_inputs.update_one(
            {"email": email, "date": yesterday},
            {"$set": {"input": stored_input}}
        )
    else:
        daily_inputs.insert_one({
            "email": email,
            "date": yesterday,
            "input": stored_input
        })

    # Build model input with rolling fatigue
    model_input = stored_input.copy()
    model_input["email"] = email
    model_input["rolling_fatigue"] = rolling

    prediction, explanation = output(model_input)

    # Check if today's prediction already exists - don't duplicate
    existing_pred = predictions.find_one({"email": email, "date": today})
    if existing_pred:
        predictions.update_one(
            {"email": email, "date": today},
            {"$set": {"prediction": prediction, "explanation": explanation}}
        )
    else:
        predictions.insert_one({
            "email": email,
            "date": today,
            "prediction": prediction,
            "explanation": explanation
        })

    return jsonify({
        "prediction": prediction,
        "explanation": explanation
    })


@app.route("/history", methods=["POST"])
def history():
    data = request.json
    today = datetime.now().strftime("%Y-%m-%d")

    today_record = predictions.find_one({
        "email": data["email"],
        "date": today
    }, {"_id": 0})

    if today_record:
        return jsonify(today_record)

    return jsonify({})


@app.route("/visualization", methods=["POST"])
def visualization():
    data = request.json

    user_data = list(daily_inputs.find(
        {"email": data["email"]}
    ).sort("date", 1))

    if len(user_data) < 2:
        return jsonify({"images": []})

    user_data = user_data[-7:]

    dates = [d["date"] for d in user_data]
    sleep = [d["input"]["noisy_sleep_hours"] for d in user_data]
    fatigue = [d["input"]["fatigue"] for d in user_data]
    intensity = [d["input"]["intensity"] for d in user_data]

    readiness = [
        d["input"]["noisy_sleep_hours"] * d["input"]["sleep_quality"]
        - d["input"]["fatigue"]
        for d in user_data
    ]

    images = []
    short_dates = [d[5:] for d in dates]

    bg_color = "#1e293b"
    text_color = "#e2e8f0"
    grid_color = "#334155"
    accent1 = "#818cf8"
    accent2 = "#f97316"
    accent3 = "#34d399"

    def style_ax(ax, fig):
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color, labelsize=10)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.title.set_fontsize(14)
        ax.title.set_fontweight("bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid_color)
        ax.spines["bottom"].set_color(grid_color)
        ax.grid(axis="y", color=grid_color, linestyle="--", alpha=0.5)

    def save_chart(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        images.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close(fig)

    # GRAPH 1: READINESS
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(short_dates, readiness, marker='o', color=accent1, linewidth=2.5, markersize=8)
    ax.fill_between(short_dates, readiness, alpha=0.15, color=accent1)
    ax.set_title("Readiness Score (Last 7 Days)")
    ax.set_ylabel("Readiness")
    style_ax(ax, fig)
    save_chart(fig)

    # GRAPH 2: SLEEP vs FATIGUE
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(short_dates, sleep, marker='o', color=accent1, linewidth=2.5, markersize=8, label="Sleep (hrs)")
    ax.plot(short_dates, fatigue, marker='s', color=accent2, linewidth=2.5, markersize=8, label="Fatigue")
    ax.set_title("Sleep vs Fatigue")
    ax.legend(facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color, fontsize=10)
    style_ax(ax, fig)
    save_chart(fig)

    # GRAPH 3: INTENSITY
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(short_dates, intensity, color=accent3, width=0.5, alpha=0.85)
    ax.set_title("Training Intensity")
    ax.set_ylabel("Intensity")
    style_ax(ax, fig)
    save_chart(fig)

    return jsonify({"images": images})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)