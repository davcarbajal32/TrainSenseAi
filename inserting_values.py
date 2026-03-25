from database import users, daily_inputs, predictions
from datetime import datetime, timedelta
import random

# Clear old test data
users.delete_many({"email": "test"})
daily_inputs.delete_many({"email": "test"})
predictions.delete_many({"email": "test"})

# Create test user
users.insert_one({"email": "test", "password": "test"})

today = datetime.now()

for i in range(30, 0, -1):
    day = (today - timedelta(days=i)).strftime("%Y-%m-%d")

    sleep = round(random.uniform(4.5, 9.0), 1)
    quality = random.randint(3, 10)
    steps = random.randint(2000, 15000)
    calories = random.randint(150, 600)
    hr = random.randint(55, 85)
    intensity = round(random.uniform(0.5, 8.0), 1)
    fatigue = random.randint(15, 85)

    daily_inputs.insert_one({
        "email": "test",
        "date": day,
        "input": {
            "noisy_sleep_hours": sleep,
            "sleep_quality": quality,
            "TotalSteps": steps,
            "TotalCalories": calories,
            "heart_rate": hr,
            "intensity": intensity,
            "fatigue": fatigue
        }
    })

print(f"Done — inserted 30 days of data for 'test' user")
print(f"Date range: {(today - timedelta(days=30)).strftime('%Y-%m-%d')} to {(today - timedelta(days=1)).strftime('%Y-%m-%d')}")