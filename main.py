"""
TrainSense - core business logic.

This module owns:
  - The MongoDB connection
  - The User class (with auth helpers)
  - The PromptBuilder class (the brain of the app - read carefully)
  - CRUD functions for profile, semesters, weekly plans, workouts,
    and athletic activities (practices/games for sports athletes)
  - Lightweight input validators

Anything Flask-specific lives in app.py. This file is importable
and runnable on its own:

    python main.py             # prints a sample prompt for inspection

Sections:
    1.  Configuration & DB
    2.  Constants & enums
    3.  Validators
    4.  User class (auth + profile)
    5.  Semester functions
    6.  Weekly plan functions
    7.  Workout functions (with sleep)
    8.  Athletic activity functions
    9.  PromptBuilder class
    10. Self-test entry point
"""

import os
import re
import bcrypt
from datetime import datetime, timezone, date, timedelta

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
from bson import ObjectId
from dotenv import load_dotenv


# =====================================================================
# 1. Configuration & DB
# =====================================================================

load_dotenv()

SECRET_KEY    = os.getenv("SECRET_KEY")
MONGO_URI     = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "trainsense")

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY not set in .env")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in .env")

mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
try:
    mongo_client.admin.command("ping")
except ConnectionFailure as e:
    raise RuntimeError(f"Could not connect to MongoDB: {e}")

db = mongo_client[MONGO_DB_NAME]

# Indexes
db.users.create_index([("email", ASCENDING)], unique=True)
db.users.create_index([("username", ASCENDING)], unique=True)
db.semesters.create_index([("user_id", ASCENDING), ("is_active", ASCENDING)])
db.workouts.create_index([("user_id", ASCENDING), ("date", DESCENDING)])
db.recommendations.create_index([("user_id", ASCENDING), ("date", DESCENDING)])
db.athletic_activities.create_index([("user_id", ASCENDING), ("date", DESCENDING)])


# =====================================================================
# 2. Constants & enums
# =====================================================================

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

ALLOWED_GOALS = {
    "hypertrophy", "strength", "aesthetics",
    "endurance", "weight_loss", "general",
    "sports_training",
}

ALLOWED_SPORTS = {
    "basketball", "soccer", "football", "volleyball",
    "baseball", "softball", "tennis", "swimming",
    "running", "lacrosse", "wrestling", "golf",
    "hockey", "rugby", "cheerleading", "dance",
    "martial_arts", "rowing", "cross_country",
    "other",
}

ALLOWED_TEAM_CONTEXTS = {"school", "club", "rec_league", "personal"}

ALLOWED_ACTIVITY_TYPES = {"practice", "game", "scrimmage"}


# =====================================================================
# 3. Validators
# =====================================================================
# All return list of error strings; empty = valid

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TIME_RE  = re.compile(r"^\d{2}:\d{2}$")


def validate_signup(data):
    if not isinstance(data, dict):
        return ["request body must be JSON"]
    errors = []
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not username or not (3 <= len(username) <= 30):
        errors.append("username must be 3-30 characters")
    if not email or not EMAIL_RE.match(email):
        errors.append("valid email is required")
    if not password or len(password) < 8:
        errors.append("password must be at least 8 characters")
    return errors


def validate_login(data):
    if not isinstance(data, dict):
        return ["request body must be JSON"]
    errors = []
    if not (data.get("email") or "").strip():
        errors.append("email is required")
    if not data.get("password"):
        errors.append("password is required")
    return errors


def validate_profile(data):
    """Validates a complete profile payload (post-signup, when user fills it in)."""
    if not isinstance(data, dict):
        return ["request body must be JSON"]
    errors = []

    goals = data.get("fitness_goals")
    if not isinstance(goals, list) or len(goals) == 0:
        errors.append("fitness_goals must be a non-empty list")
    else:
        bad = [g for g in goals if g not in ALLOWED_GOALS]
        if bad:
            errors.append(f"unknown goals: {bad}")

    # Sport fields - only required when sports_training is selected
    if isinstance(goals, list) and "sports_training" in goals:
        sport = data.get("sport")
        if sport not in ALLOWED_SPORTS:
            errors.append(f"sport required when sports_training selected; allowed: {sorted(ALLOWED_SPORTS)}")
        if sport == "other":
            sport_other = (data.get("sport_other") or "").strip()
            if not sport_other or len(sport_other) > 50:
                errors.append("sport_other required (1-50 chars) when sport is 'other'")
        team_ctx = data.get("team_context")
        if team_ctx not in ALLOWED_TEAM_CONTEXTS:
            errors.append(f"team_context required: one of {sorted(ALLOWED_TEAM_CONTEXTS)}")

    budget = data.get("budget_monthly_usd")
    if not isinstance(budget, int) or not (0 <= budget <= 1000):
        errors.append("budget_monthly_usd must be an integer 0-1000")

    energy = data.get("energy_levels")
    if not isinstance(energy, dict):
        errors.append("energy_levels required: {morning, afternoon, evening} each 1-10")
    else:
        for slot in ("morning", "afternoon", "evening"):
            v = energy.get(slot)
            if not isinstance(v, int) or not (1 <= v <= 10):
                errors.append(f"energy_levels.{slot} must be an integer 1-10")

    # Optional injuries
    inj = data.get("injuries_limitations")
    if inj is not None and (not isinstance(inj, str) or len(inj) > 500):
        errors.append("injuries_limitations must be a string under 500 characters")

    loc = data.get("location")
    if not isinstance(loc, dict) or not (loc.get("city") or "").strip():
        errors.append("location.city is required")

    return errors


def _validate_time_block(block, *, with_label):
    errors = []
    if with_label:
        label = (block.get("label") or "").strip()
        if not label or len(label) > 100:
            errors.append("label must be 1-100 characters")
    if block.get("day_of_week") not in DAYS:
        errors.append(f"day_of_week must be one of {DAYS}")
    if not TIME_RE.match(block.get("start_time") or ""):
        errors.append("start_time must be HH:MM")
    if not TIME_RE.match(block.get("end_time") or ""):
        errors.append("end_time must be HH:MM")
    return errors


def validate_semester(data):
    if not isinstance(data, dict):
        return ["request body must be JSON"]
    errors = []
    if not (data.get("label") or "").strip():
        errors.append("label is required")
    for field in ("start_date", "end_date"):
        if not data.get(field):
            errors.append(f"{field} is required (ISO format)")

    classes = data.get("classes")
    if not isinstance(classes, list):
        errors.append("classes must be a list")
    else:
        for i, c in enumerate(classes):
            for e in _validate_time_block(c, with_label=True):
                errors.append(f"classes[{i}]: {e}")
    return errors


def validate_weekly_plan(data):
    if not isinstance(data, dict):
        return ["request body must be JSON"]
    errors = []
    if not data.get("week_start"):
        errors.append("week_start is required (ISO date of Monday)")

    work_blocks = data.get("work_blocks", [])
    if not isinstance(work_blocks, list):
        errors.append("work_blocks must be a list")
    else:
        for i, b in enumerate(work_blocks):
            for e in _validate_time_block(b, with_label=False):
                errors.append(f"work_blocks[{i}]: {e}")

    study = data.get("study_hours")
    if not isinstance(study, dict):
        errors.append("study_hours must be an object with keys mon-sun")
    else:
        for d in DAYS:
            v = study.get(d)
            if not isinstance(v, (int, float)) or not (0 <= v <= 24):
                errors.append(f"study_hours.{d} must be a number 0-24")
    return errors


def validate_workout(data):
    if not isinstance(data, dict):
        return ["request body must be JSON"]
    errors = []

    # Sleep is always asked (even on rest days)
    sleep = data.get("sleep_hours")
    if sleep is not None and (not isinstance(sleep, (int, float)) or not (0 <= sleep <= 14)):
        errors.append("sleep_hours must be a number 0-14")

    if not isinstance(data.get("did_workout"), bool):
        errors.append("did_workout must be true or false")
        return errors

    if data["did_workout"]:
        if not (data.get("workout_type") or "").strip():
            errors.append("workout_type required when did_workout is true")
        d = data.get("duration_minutes")
        if not isinstance(d, int) or not (1 <= d <= 600):
            errors.append("duration_minutes must be an integer 1-600")
        i = data.get("intensity")
        if not isinstance(i, int) or not (1 <= i <= 10):
            errors.append("intensity must be an integer 1-10")

    # followed_plan is optional. When present, must be a bool. Only meaningful
    # when did_workout is true AND a recommendation exists for that day.
    fp = data.get("followed_plan")
    if fp is not None and not isinstance(fp, bool):
        errors.append("followed_plan must be true or false")

    notes = data.get("notes")
    if notes is not None and (not isinstance(notes, str) or len(notes) > 1000):
        errors.append("notes must be a string under 1000 characters")
    return errors


def validate_athletic_activity(data):
    if not isinstance(data, dict):
        return ["request body must be JSON"]
    errors = []

    if not data.get("date"):
        errors.append("date is required (ISO date)")
    else:
        # Cap backfill at 7 days (per design decision)
        try:
            dt = _parse_date(data["date"])
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            if dt > today:
                errors.append("date cannot be in the future")
            elif (today - dt).days > 7:
                errors.append("date cannot be more than 7 days in the past")
        except Exception:
            errors.append("date must be a valid ISO date")

    if data.get("activity_type") not in ALLOWED_ACTIVITY_TYPES:
        errors.append(f"activity_type must be one of {sorted(ALLOWED_ACTIVITY_TYPES)}")

    d = data.get("duration_minutes")
    if not isinstance(d, int) or not (1 <= d <= 600):
        errors.append("duration_minutes must be an integer 1-600")

    desc = (data.get("description") or "").strip()
    if not desc or len(desc) > 500:
        errors.append("description required (1-500 chars)")

    i = data.get("intensity")
    if not isinstance(i, int) or not (1 <= i <= 10):
        errors.append("intensity must be an integer 1-10")

    return errors


# =====================================================================
# 4. User class
# =====================================================================

class User:
    """Wraps a MongoDB user document with auth + profile behavior.
    Compatible with Flask-Login."""

    def __init__(self, doc):
        self._doc = doc
        self.id = str(doc["_id"])
        self.username = doc["username"]
        self.email = doc["email"]
        self.profile = doc.get("profile", {})
        self.profile_completed = doc.get("profile_completed", False)

    # ----- Lookups -----

    @staticmethod
    def find_by_id(user_id):
        try:
            doc = db.users.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None
        return User(doc) if doc else None

    @staticmethod
    def find_by_email(email):
        doc = db.users.find_one({"email": (email or "").lower().strip()})
        return User(doc) if doc else None

    # ----- Signup / login -----

    @staticmethod
    def create(username, email, password):
        username = username.strip()
        email = email.strip().lower()

        if db.users.find_one({"$or": [{"email": email}, {"username": username}]}):
            return None, "username or email already in use"

        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        doc = {
            "username": username,
            "email": email,
            "password_hash": pw_hash,
            "created_at": datetime.now(timezone.utc),
            "profile_completed": False,
            "profile": {
                "fitness_goals": None,
                "sport": None,
                "sport_other": None,
                "team_context": None,
                "budget_monthly_usd": None,
                "energy_levels": None,
                "injuries_limitations": None,
                "location": None,
            },
            "current_week": None,
            "weekly_plan_history": [],
        }
        result = db.users.insert_one(doc)
        doc["_id"] = result.inserted_id
        return User(doc), None

    @staticmethod
    def authenticate(email, password):
        user = User.find_by_email(email)
        # Constant-time defense against user enumeration via timing
        dummy = b"$2b$12$" + b"x" * 53
        stored = user.password_hash if user else dummy
        if not bcrypt.checkpw(password.encode("utf-8"), stored) or user is None:
            return None, "invalid credentials"
        return user, None

    # ----- Profile -----

    def update_profile(self, profile_data):
        is_athlete = "sports_training" in (profile_data.get("fitness_goals") or [])

        # Geocode city to (lat, lon) so weather lookups work.
        # Open-Meteo is free and doesn't need a key. If geocoding fails
        # (offline, weird city name), we silently fall back to null lat/lon
        # and Claude prompts still work - they just say "weather unavailable".
        from weather import geocode_city
        city_input = (profile_data.get("location") or {}).get("city", "")
        geocoded = geocode_city(city_input)
        if geocoded:
            lat, lon, _resolved = geocoded
        else:
            lat, lon = None, None

        new_profile = {
            "fitness_goals": profile_data["fitness_goals"],
            "sport": profile_data.get("sport") if is_athlete else None,
            "sport_other": (profile_data.get("sport_other") or "").strip() or None if is_athlete else None,
            "team_context": profile_data.get("team_context") if is_athlete else None,
            "budget_monthly_usd": profile_data["budget_monthly_usd"],
            "energy_levels": profile_data["energy_levels"],
            "injuries_limitations": (profile_data.get("injuries_limitations") or "").strip() or None,
            "location": {
                "city": profile_data["location"]["city"].strip(),
                "lat": lat,
                "lon": lon,
            },
        }
        db.users.update_one(
            {"_id": ObjectId(self.id)},
            {"$set": {"profile": new_profile, "profile_completed": True}},
        )
        self._doc = db.users.find_one({"_id": ObjectId(self.id)})
        self.profile = self._doc["profile"]
        self.profile_completed = True

    # ----- Flask-Login interface -----

    @property
    def is_authenticated(self): return True
    @property
    def is_active(self): return True
    @property
    def is_anonymous(self): return False
    def get_id(self): return self.id

    @property
    def password_hash(self):
        return self._doc["password_hash"]


# =====================================================================
# 5. Semester functions
# =====================================================================

def _parse_date(s):
    """Accepts ISO string, date, or datetime. Returns naive UTC datetime
    (matches how MongoDB stores datetimes, so all comparisons stay consistent)."""
    if isinstance(s, datetime):
        return s.replace(tzinfo=None) if s.tzinfo else s
    if isinstance(s, date):
        return datetime(s.year, s.month, s.day)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def create_semester(user_id, label, start_date, end_date, classes):
    """Sets all other semesters for this user to inactive, then creates this one as active."""
    uid = ObjectId(user_id)
    db.semesters.update_many({"user_id": uid}, {"$set": {"is_active": False}})

    doc = {
        "user_id": uid,
        "label": label.strip(),
        "start_date": _parse_date(start_date),
        "end_date": _parse_date(end_date),
        "is_active": True,
        "classes": [
            {
                "label": c["label"].strip(),
                "day_of_week": c["day_of_week"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
            }
            for c in classes
        ],
        "created_at": datetime.now(timezone.utc),
    }
    result = db.semesters.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_active_semester(user_id):
    """Returns the active semester WITH the computed `is_currently_active` field.
    The stored is_active flag may lag reality - this function tells the truth."""
    sem = db.semesters.find_one({
        "user_id": ObjectId(user_id),
        "is_active": True,
    })
    if not sem:
        return None
    sem["is_currently_active"] = is_semester_currently_active(sem)
    return sem


def is_semester_currently_active(sem):
    """A semester is currently active iff today is within its date range.
    MongoDB strips timezone info on datetimes by default, so we compare in
    naive UTC to avoid offset-aware vs offset-naive TypeError."""
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = sem["start_date"]
    end = sem["end_date"]
    # Strip tzinfo if present (Mongo returns naive; in-memory may be aware)
    if start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.replace(tzinfo=None)
    return start <= today <= end


def deactivate_past_semesters(user_id):
    """Auto-flip is_active=false on any of this user's semesters whose end_date has passed.
    Called at login. Cheap, idempotent. Uses naive UTC to match how Mongo stores datetimes."""
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    db.semesters.update_many(
        {"user_id": ObjectId(user_id), "is_active": True, "end_date": {"$lt": today}},
        {"$set": {"is_active": False}},
    )


# =====================================================================
# 6. Weekly plan functions
# =====================================================================

def save_weekly_plan(user_id, week_start, work_blocks, study_hours):
    """Archives the previous current_week into history and sets the new one."""
    uid = ObjectId(user_id)
    user_doc = db.users.find_one({"_id": uid})
    if not user_doc:
        raise ValueError(f"user {user_id} not found")

    new_plan = {
        "week_start": _parse_date(week_start),
        "work_blocks": [
            {
                "day_of_week": b["day_of_week"],
                "start_time": b["start_time"],
                "end_time": b["end_time"],
            }
            for b in work_blocks
        ],
        "study_hours": {d: float(study_hours[d]) for d in DAYS},
    }

    update = {"$set": {"current_week": new_plan}}
    previous = user_doc.get("current_week")
    if previous:
        update["$push"] = {"weekly_plan_history": previous}

    db.users.update_one({"_id": uid}, update)
    return new_plan


# =====================================================================
# 7. Workout functions (with sleep)
# =====================================================================

def log_workout(user_id, date_in, did_workout, sleep_hours=None,
                workout_type=None, duration_minutes=None,
                intensity=None, notes=None, followed_plan=None):
    """Inserts (or replaces) a workout log for the given date."""
    uid = ObjectId(user_id)
    date_dt = _parse_date(date_in) if isinstance(date_in, str) else date_in

    doc = {
        "user_id": uid,
        "date": date_dt,
        "sleep_hours": float(sleep_hours) if sleep_hours is not None else None,
        "did_workout": did_workout,
        "workout_type": (workout_type.strip() if workout_type else None) if did_workout else None,
        "duration_minutes": duration_minutes if did_workout else None,
        "intensity": intensity if did_workout else None,
        "followed_plan": followed_plan if did_workout else None,
        "notes": (notes.strip() if notes else None) or None,
        "logged_at": datetime.now(timezone.utc),
    }

    db.workouts.replace_one(
        {"user_id": uid, "date": date_dt},
        doc,
        upsert=True,
    )
    return doc


def get_recent_workouts(user_id, limit=4, before=None):
    """Returns last `limit` logs strictly before `before` (defaults to now), newest first."""
    uid = ObjectId(user_id)
    if before is None:
        before = datetime.utcnow()
    elif isinstance(before, str):
        before = _parse_date(before)
    elif before.tzinfo is not None:
        before = before.replace(tzinfo=None)

    return list(db.workouts.find({
        "user_id": uid,
        "date": {"$lt": before},
    }).sort("date", -1).limit(limit))


# =====================================================================
# 8. Athletic activity functions
# =====================================================================

def log_athletic_activity(user_id, date_in, activity_type, duration_minutes,
                           description, intensity):
    """Insert or replace an athletic activity for a given (user, date, activity_type).
    A user might have practice AND a game on the same day — distinct activity_type
    means we keep them separate."""
    uid = ObjectId(user_id)
    date_dt = _parse_date(date_in) if isinstance(date_in, str) else date_in

    doc = {
        "user_id": uid,
        "date": date_dt,
        "activity_type": activity_type,
        "duration_minutes": duration_minutes,
        "description": description.strip(),
        "intensity": intensity,
        "logged_at": datetime.now(timezone.utc),
    }

    db.athletic_activities.replace_one(
        {"user_id": uid, "date": date_dt, "activity_type": activity_type},
        doc,
        upsert=True,
    )
    return doc


def get_recent_athletic_activities(user_id, limit=10, before=None):
    uid = ObjectId(user_id)
    if before is None:
        before = datetime.utcnow()
    elif isinstance(before, str):
        before = _parse_date(before)
    elif before.tzinfo is not None:
        before = before.replace(tzinfo=None)

    return list(db.athletic_activities.find({
        "user_id": uid,
        "date": {"$lt": before},
    }).sort("date", -1).limit(limit))


# =====================================================================
# 8b. Recommendation orchestration (Claude calls + storage)
# =====================================================================
# These functions stitch together the prompt builder, the Claude client,
# the weather lookup, and MongoDB storage. They're the public API the
# routes call to actually generate recommendations.

def store_daily_recommendation(user_id, target_date, parsed_response, raw_text=None,
                                context_used=None, model="claude-sonnet-4-6"):
    """Save one day's recommendation to the recommendations collection.
    Replaces any existing recommendation for that (user, date) - we only
    keep the latest version per day."""
    uid = ObjectId(user_id)
    target_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if target_date.tzinfo is not None:
        target_date = target_date.replace(tzinfo=None)

    p = parsed_response or {}
    doc = {
        "user_id":          uid,
        "date":             target_date,
        "generated_at":     datetime.utcnow(),
        "model":            model,
        "is_rest_day":      bool(p.get("is_rest_day")),
        "recommended_time": p.get("recommended_time"),
        "workout_type":     p.get("workout_type"),
        "intensity":        p.get("intensity"),
        "duration_minutes": p.get("duration_minutes"),
        "reasoning":        p.get("reasoning") or "",
        "context_used":     context_used or {},
        "raw_response":     raw_text,
    }

    db.recommendations.replace_one(
        {"user_id": uid, "date": target_date},
        doc,
        upsert=True,
    )
    return doc


def compute_deviations_for_week(user_id, monday, today):
    """For each past day this week (monday through yesterday), compare what was
    planned to what was logged. Returns a human-readable summary string suitable
    for inclusion in the prompt, or None if there's nothing notable.

    Examples of deviations we capture:
      - Plan said workout, user took rest
      - Plan said rest, user worked out
      - Plan said hypertrophy, user did cardio (followed_plan=false)
      - User slept poorly relative to plan
    """
    uid = ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id

    if today.tzinfo is not None:
        today = today.replace(tzinfo=None)
    if monday.tzinfo is not None:
        monday = monday.replace(tzinfo=None)

    # Pull all recommendations and logs for monday through yesterday
    yesterday = today - timedelta(days=1)
    if yesterday < monday:
        return None  # it's still Monday, no past days yet

    recs = list(db.recommendations.find({
        "user_id": uid,
        "date": {"$gte": monday, "$lte": yesterday},
    }).sort("date", 1))
    logs = list(db.workouts.find({
        "user_id": uid,
        "date": {"$gte": monday, "$lte": yesterday},
    }).sort("date", 1))
    logs_by_date = {w["date"].date(): w for w in logs}

    if not recs and not logs:
        return None

    lines = []
    for r in recs:
        d = r["date"].date()
        log = logs_by_date.get(d)
        date_str = r["date"].strftime("%a %b %d")
        plan_desc = "rest day" if r.get("is_rest_day") else (r.get("workout_type") or "workout (unspecified)")

        if not log:
            lines.append(f"{date_str}: planned {plan_desc} - no check-in logged (status unclear)")
            continue

        sleep = log.get("sleep_hours")
        sleep_note = f", slept {sleep}hrs" if sleep is not None else ""

        if r.get("is_rest_day"):
            if log.get("did_workout"):
                lines.append(
                    f"{date_str}: planned REST, but user worked out "
                    f"({log.get('workout_type', 'unspecified')}, "
                    f"{log.get('duration_minutes', '?')}min, "
                    f"intensity {log.get('intensity', '?')}/10){sleep_note}"
                )
            else:
                lines.append(f"{date_str}: planned rest, user rested as planned{sleep_note}")
        else:
            if not log.get("did_workout"):
                lines.append(f"{date_str}: planned {plan_desc}, user took rest instead{sleep_note}")
            else:
                followed = log.get("followed_plan")
                actual = log.get("workout_type", "unspecified")
                int_str = f"intensity {log.get('intensity', '?')}/10"
                dur_str = f"{log.get('duration_minutes', '?')}min"
                if followed is True:
                    lines.append(
                        f"{date_str}: planned {plan_desc}, user FOLLOWED plan "
                        f"({actual}, {dur_str}, {int_str}){sleep_note}"
                    )
                elif followed is False:
                    lines.append(
                        f"{date_str}: planned {plan_desc}, but user did DIFFERENT workout "
                        f"({actual}, {dur_str}, {int_str}){sleep_note}"
                    )
                else:
                    lines.append(
                        f"{date_str}: planned {plan_desc}, user worked out "
                        f"({actual}, {dur_str}, {int_str}, plan-match unrecorded){sleep_note}"
                    )

    # Also capture days that have logs but no recommendation (rare - generated mid-week)
    for d, log in logs_by_date.items():
        if any(r["date"].date() == d for r in recs):
            continue
        date_str = log["date"].strftime("%a %b %d")
        if log.get("did_workout"):
            lines.append(
                f"{date_str}: no plan on file, user worked out "
                f"({log.get('workout_type', 'unspecified')}, "
                f"{log.get('duration_minutes', '?')}min, "
                f"intensity {log.get('intensity', '?')}/10)"
            )

    return "\n".join(lines) if lines else None


def generate_weekly_recommendations(user_id, week_start=None):
    """Generate this week's plan from TODAY through Sunday, leaving past days'
    recommendations untouched (so the calendar still shows the original plan
    vs what the user actually did, with green/yellow markers).

    The prompt includes:
      - Recent workout logs (always)
      - An explicit 'deviations summary' showing planned vs actual for past days

    Returns the parsed week response. Raises RuntimeError on Claude failure.
    """
    from claude import call_claude
    from weather import get_today_weather

    uid = ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    if week_start is None:
        week_start = today - timedelta(days=today.weekday())
    elif week_start.tzinfo is not None:
        week_start = week_start.replace(tzinfo=None)

    # Only regenerate from `regen_start` onward. If today is Monday, that's the
    # whole week. If today is Wednesday, we leave Mon and Tue alone and
    # regenerate Wed through Sun.
    regen_start = max(today, week_start)

    # Compute the past-week deviation summary (if we have past days this week)
    deviation_summary = compute_deviations_for_week(uid, week_start, today)

    pb = PromptBuilder()
    user_doc, semester_doc, recent_w, recent_a = pb.gather_context(uid, target_date=regen_start)

    profile = (user_doc or {}).get("profile") or {}
    loc = profile.get("location") or {}
    weather = get_today_weather(loc.get("lat"), loc.get("lon")) if loc.get("lat") else None

    sys_p, user_p = pb.build_weekly(
        user_doc, semester_doc, recent_w, recent_a, weather,
        regen_start,
        full_week_start=week_start,
        deviation_summary=deviation_summary,
    )
    raw, parsed = call_claude(sys_p, user_p)

    if not parsed or not isinstance(parsed.get("days"), list):
        raise RuntimeError(f"Claude returned unparseable weekly response: {raw[:300]}")

    context_used = {
        "weather": weather,
        "recent_workouts_count": len(recent_w),
        "week_load": _summarize_week_load(user_doc),
        "regen_start": regen_start.date().isoformat(),
        "deviation_summary": deviation_summary,
    }

    # Store each day as its own recommendation - but ONLY for dates >= regen_start.
    # Claude might return all 7 days; we ignore past days to preserve the original plan.
    stored = []
    skipped_past = 0
    for day_entry in parsed["days"]:
        try:
            d = _parse_date(day_entry["date"])
        except Exception:
            continue
        if d < regen_start:
            skipped_past += 1
            continue
        doc = store_daily_recommendation(
            uid, d, day_entry, raw_text=None,
            context_used=context_used,
        )
        stored.append(doc)

    return {
        "week_summary": parsed.get("week_summary", ""),
        "days_stored": len(stored),
        "past_days_preserved": skipped_past,
        "raw_text": raw,
    }


def revise_today_recommendation(user_id, target_date=None, deviation_reason="user requested"):
    """Generate a revised recommendation for ONE day based on new info
    (poor sleep, schedule change, etc.)."""
    from claude import call_claude
    from weather import get_today_weather

    uid = ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id

    if target_date is None:
        target_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    elif target_date.tzinfo is not None:
        target_date = target_date.replace(tzinfo=None)

    pb = PromptBuilder()
    user_doc, semester_doc, recent_w, recent_a = pb.gather_context(uid, target_date=target_date)

    profile = (user_doc or {}).get("profile") or {}
    loc = profile.get("location") or {}
    weather = get_today_weather(loc.get("lat"), loc.get("lon")) if loc.get("lat") else None

    # Get the original recommendation if one exists
    target_dt = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    original = db.recommendations.find_one({"user_id": uid, "date": target_dt})

    sys_p, user_p = pb.build_daily_revision(
        user_doc, semester_doc, recent_w, recent_a, weather, target_date,
        original, deviation_reason,
    )
    raw, parsed = call_claude(sys_p, user_p)

    if not parsed:
        raise RuntimeError(f"Claude returned unparseable revision response: {raw[:300]}")

    context_used = {
        "weather": weather,
        "recent_workouts_count": len(recent_w),
        "deviation_reason": deviation_reason,
        "revised_from_original": bool(original),
    }
    doc = store_daily_recommendation(
        uid, target_date, parsed, raw_text=raw, context_used=context_used,
    )
    return doc


def detect_deviation(user_id, target_date=None):
    """Compare last night's check-in to what the original plan assumed.
    Returns a string describing the deviation, or None if no revision needed.

    Triggers a revision if any of:
      - Last night's sleep < 6 hours
      - Yesterday's intensity was 9 or 10
      - The user's plan was a workout but they logged a rest day
    """
    uid = ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id
    if target_date is None:
        target_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    elif target_date.tzinfo is not None:
        target_date = target_date.replace(tzinfo=None)

    yesterday = target_date - timedelta(days=1)

    yesterday_log = db.workouts.find_one({"user_id": uid, "date": yesterday})
    if not yesterday_log:
        return None

    reasons = []
    if yesterday_log.get("sleep_hours") is not None and yesterday_log["sleep_hours"] < 6:
        reasons.append(f"User slept {yesterday_log['sleep_hours']} hours last night (under 6 - sleep deficit, drop intensity tier per Rule #2)")
    if yesterday_log.get("intensity") and yesterday_log["intensity"] >= 9:
        reasons.append(f"Yesterday's intensity was {yesterday_log['intensity']}/10 - high systemic fatigue likely")

    yesterday_rec = db.recommendations.find_one({"user_id": uid, "date": yesterday})
    if yesterday_rec and not yesterday_rec.get("is_rest_day") and not yesterday_log.get("did_workout"):
        reasons.append("Yesterday's plan was a workout but user took a rest day - shift volume forward in the week")

    return "; ".join(reasons) if reasons else None


def _summarize_week_load(user_doc):
    """Crude summary of the week's academic load."""
    cw = user_doc.get("current_week") if user_doc else None
    if not cw:
        return None
    total_study = sum((cw.get("study_hours") or {}).values())
    total_work = len(cw.get("work_blocks", []))
    if total_study >= 25 or total_work >= 5:
        return f"heavy week ({total_study:.0f} study hrs, {total_work} work blocks)"
    if total_study >= 15:
        return f"moderate week ({total_study:.0f} study hrs)"
    return f"light week ({total_study:.0f} study hrs)"


# =====================================================================
# 9. PromptBuilder class
# =====================================================================
# THIS IS THE BRAIN. Read carefully, modify freely.
#
# The PromptBuilder produces prompts for THREE call types:
#   build_daily()            - one day's recommendation
#   build_weekly()           - Sunday's full-week generation
#   build_daily_revision()   - small revision when something changed mid-week
#
# All three share the same SYSTEM PROMPT (the coach persona). They differ
# only in the user prompt's task section.

class PromptBuilder:

    HISTORY_DEPTH = 4   # last N days of workouts/activities to include

    # =================================================================
    # System prompt - the coach persona
    # =================================================================

    SYSTEM_PROMPT = (
        "You are a senior strength and conditioning coach with 15+ years of experience working "
        "with collegiate athletes, recreational lifters, and student-athletes balancing academics "
        "with training. Your credentials are equivalent to CSCS (Certified Strength and Conditioning "
        "Specialist), and you reason from contemporary sports science literature, including:\n\n"

        "- Schoenfeld's research on hypertrophy mechanisms: mechanical tension as the primary driver, "
        "the role of training volume (sets per muscle per week), proximity to failure, and frequency "
        "distribution.\n"
        "- Helms and Israetel's evidence-based programming for natural lifters: MEV/MAV/MRV volume "
        "landmarks, autoregulation via RPE.\n"
        "- Periodization frameworks: linear, daily undulating (DUP), block periodization, and when "
        "each is appropriate.\n"
        "- Recovery science: sleep architecture's role in muscle protein synthesis (deep sleep is when "
        "GH is released and MPS peaks), the cost of chronic sleep debt on neural recovery, "
        "parasympathetic vs sympathetic balance.\n"
        "- The SAID principle (Specific Adaptation to Imposed Demands): training adaptations are "
        "highly specific to the imposed demand - strength work builds strength, hypertrophy work "
        "builds size, sport-specific work builds sport performance.\n"
        "- Concurrent training interference: how endurance work attenuates strength/hypertrophy gains "
        "when programmed poorly, and how to sequence sessions when both are required.\n\n"

        "You apply these principles rigorously to every recommendation. You DO NOT recommend training "
        "that doesn't serve the athlete's stated goal:\n"
        "- Hypertrophy athletes get controlled tempo work, moderate-rep ranges (6-15), proximity-to-"
        "failure protocols, and 10-20 working sets per muscle per week. Not 'fun variety' circuit training.\n"
        "- Strength athletes get heavy compound work in the 1-5 rep range, longer rest periods (3-5 min), "
        "and progressive loading. Not yoga, not random aesthetic work.\n"
        "- Aesthetics athletes get hypertrophy-style training with extra volume on visually-prioritized "
        "muscles (lats, delts, arms, glutes, calves).\n"
        "- Endurance athletes get appropriately periodized cardio (zone 2 base, threshold work, intervals) "
        "with strength as a supporting modality.\n"
        "- Weight loss athletes get a mix balancing energy expenditure with muscle preservation - "
        "full-body resistance training plus zone 2 cardio.\n"
        "- Sports-training athletes get sport-specific power, conditioning, and injury-prevention work, "
        "with gym sessions designed AROUND practice and game schedules.\n\n"

        "You vary EXERCISES (different rep schemes, exercise selection within the same movement pattern, "
        "undulating intensity within the week) to prevent plateaus and boredom - but you NEVER vary the "
        "TRAINING MODALITY away from what serves the goal. A strength athlete bored of squats gets front "
        "squats or pause squats, not Pilates.\n\n"

        "You think in terms of weekly volume, not just today's session. A 'good Monday' is one that fits "
        "inside a coherent week.\n\n"

        "HARD RULES (these override everything else):\n"
        "1. Never recommend a workout that overlaps with class, work, or athletic practice/game commitments.\n"
        "2. If the athlete slept under 6 hours last night, drop intensity by one tier (high to medium, "
        "medium to low) and prioritize lower neurological cost (avoid heavy compounds, max-effort work, "
        "complex skill-based movements).\n"
        "3. If recent intensity has been 8+/10 for 3+ consecutive days, mandate a deload session or full "
        "rest. Overreaching is real.\n"
        "4. For hypertrophy goals: schedule lifting 2-4 hours before typical bedtime when possible, so "
        "core temperature can drop before sleep onset. Morning chronotypes train morning.\n"
        "5. For sports-training athletes: avoid heavy lower-body lifting within 24-48 hours of a known "
        "game/competition. Treat practice days as partial training stress; don't double up high-intensity "
        "gym work on heavy practice days.\n"
        "6. If the athlete has logged injuries or limitations, route around them strictly. A 'shoulder "
        "limitation' means no overhead pressing, dips, or behind-neck work. A 'lower back' limitation "
        "means no deadlifts, heavy bent-over rows, or loaded spinal flexion.\n"
        "7. Respect the athlete's available equipment. Do not recommend cable crossovers if they only "
        "have access to a campus gym with dumbbells.\n\n"

        "Output format is strictly JSON, no markdown, no preamble, no explanation outside the JSON. "
        "Reasoning must reference SPECIFIC principles you applied to THIS athlete on THIS day - not "
        "vague platitudes."
    )

    # =================================================================
    # Public API
    # =================================================================

    def build_daily(self, user_doc, semester_doc, recent_workouts,
                    recent_activities, weather, target_date=None):
        """Returns (system, user_prompt) for ONE day's recommendation."""
        if target_date is None:
            target_date = datetime.utcnow()
        elif target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)

        sections = self._common_user_sections(
            user_doc, semester_doc, recent_workouts, recent_activities,
            weather, target_date, target_date,
        )
        sections.append(self._daily_task_section(target_date))
        return self.SYSTEM_PROMPT, "\n\n".join(sections)

    def build_weekly(self, user_doc, semester_doc, recent_workouts,
                     recent_activities, weather, week_start,
                     full_week_start=None, deviation_summary=None):
        """Returns (system, user_prompt) for weekly generation.

        Args:
            week_start: First date to generate FOR (inclusive). If we're mid-week
                and only regenerating from today onward, this is today.
            full_week_start: The actual Monday of the calendar week, used for
                showing all 7 days' commitments to Claude. If None, equals week_start.
            deviation_summary: Optional human-readable summary of what was planned
                vs actually done for past days this week. Empowers Claude to
                course-correct the upcoming days.
        """
        if full_week_start is None:
            full_week_start = week_start
        week_end = full_week_start + timedelta(days=6)

        sections = self._common_user_sections(
            user_doc, semester_doc, recent_workouts, recent_activities,
            weather, week_start, week_end,
        )

        # Past-days deviation summary - only included when there's something to report.
        # This is THE critical signal for partial regenerations: it tells Claude
        # exactly where the user diverged from the original plan.
        if deviation_summary:
            sections.append(
                "=== PAST DAYS THIS WEEK (planned vs actual) ===\n"
                f"{deviation_summary}\n\n"
                "These past days are LOCKED — do not generate recommendations for them. "
                "Use them as the source of truth for what the athlete actually did, "
                "and let that override the original plan when you decide what comes next. "
                "If they did extra volume already, scale back. If they skipped a session, "
                "consider whether to redistribute or accept the lost volume. If they did "
                "a different workout type, balance the week's overall volume around what "
                "actually happened, not what was planned."
            )

        sections.append(self._weekly_constraints_section(user_doc, semester_doc, full_week_start))
        sections.append(self._weekly_task_section(week_start, full_week_start))
        return self.SYSTEM_PROMPT, "\n\n".join(sections)

    def build_daily_revision(self, user_doc, semester_doc, recent_workouts,
                             recent_activities, weather, target_date,
                             original_recommendation, deviation_reason):
        """Returns (system, user_prompt) when something changed mid-week and we need
        to revise today's plan. Smaller prompt - we don't re-derive the whole week."""
        sections = [
            f"=== DATE ===\n{target_date.strftime('%A, %B %d, %Y')}",
            self._profile_section(user_doc),
            self._today_constraints_section(user_doc, semester_doc, recent_activities, target_date),
            self._recovery_section(recent_workouts, recent_activities),
            self._original_plan_section(original_recommendation),
            f"=== WHAT CHANGED ===\n{deviation_reason}",
            self._daily_revision_task_section(target_date),
        ]
        return self.SYSTEM_PROMPT, "\n\n".join(sections)

    def gather_context(self, user_id, target_date=None):
        """Pull everything needed to build a daily prompt. Returns the splattable args."""
        if target_date is None:
            target_date = datetime.utcnow()
        elif target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)

        uid = ObjectId(user_id) if not isinstance(user_id, ObjectId) else user_id
        user_doc = db.users.find_one({"_id": uid})
        semester_doc = get_active_semester(uid)

        cutoff = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        recent_w = list(db.workouts.find({
            "user_id": uid, "date": {"$lt": cutoff},
        }).sort("date", -1).limit(self.HISTORY_DEPTH))
        recent_a = list(db.athletic_activities.find({
            "user_id": uid, "date": {"$lt": cutoff},
        }).sort("date", -1).limit(self.HISTORY_DEPTH))

        return user_doc, semester_doc, recent_w, recent_a

    # =================================================================
    # Section builders (composed by build_daily / build_weekly / revision)
    # =================================================================

    def _common_user_sections(self, user_doc, semester_doc, recent_workouts,
                               recent_activities, weather, range_start, range_end):
        return [
            f"=== DATE ===\n{range_start.strftime('%A, %B %d, %Y')}"
            + (f" through {range_end.strftime('%A, %B %d, %Y')}" if range_end != range_start else ""),
            self._profile_section(user_doc),
            self._today_constraints_section(user_doc, semester_doc, recent_activities, range_start),
            self._recovery_section(recent_workouts, recent_activities),
            self._weather_section(weather, user_doc),
        ]

    def _profile_section(self, user_doc):
        profile = user_doc.get("profile", {})
        lines = ["=== ATHLETE PROFILE ==="]

        goals = profile.get("fitness_goals") or []
        lines.append(f"Primary goals: {self._goals_phrase(goals)}")

        if "sports_training" in goals:
            sport = profile.get("sport") or "unspecified"
            if sport == "other" and profile.get("sport_other"):
                sport = profile["sport_other"]
            ctx = profile.get("team_context") or "personal"
            ctx_phrase = {
                "school": "school team", "club": "club team",
                "rec_league": "recreational league", "personal": "personal/casual",
            }.get(ctx, ctx)
            lines.append(f"Sport: {sport} ({ctx_phrase})")

        lines.append(f"Equipment access: {self._budget_phrase(profile.get('budget_monthly_usd'))}")
        lines.append(f"Energy pattern: {self._energy_phrase(profile.get('energy_levels'))}")

        inj = profile.get("injuries_limitations")
        if inj:
            lines.append(f"Injuries / limitations: {inj}")
        else:
            lines.append("Injuries / limitations: none reported")

        loc = profile.get("location") or {}
        if loc.get("city"):
            lines.append(f"Location: {loc['city']}")

        return "\n".join(lines)

    def _today_constraints_section(self, user_doc, semester_doc, recent_activities, target_date):
        dow = self._day_of_week(target_date)
        target_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        current_week = user_doc.get("current_week") or {}

        classes = self._format_classes(semester_doc, dow)
        work    = self._format_work(current_week.get("work_blocks", []), dow)
        study   = self._format_study(current_week.get("study_hours"), dow)

        # Athletic activities for today's date specifically
        todays_activities = [a for a in recent_activities
                              if a["date"].date() == target_day.date()]
        if todays_activities:
            ath_lines = []
            for a in todays_activities:
                ath_lines.append(
                    f"{a['activity_type']}: {a['duration_minutes']} min, "
                    f"intensity {a['intensity']}/10 - {a['description']}"
                )
            athletic = "; ".join(ath_lines)
        else:
            athletic = "none scheduled"

        # Compute available training windows from class+work+activity blocks for today
        windows = self._available_windows(semester_doc, current_week, todays_activities, dow)

        return (
            "=== TODAY'S COMMITMENTS ===\n"
            f"Classes: {classes}\n"
            f"Work: {work}\n"
            f"Study load: {study}\n"
            f"Athletic practice/games today: {athletic}\n"
            f"Available training windows: {windows}"
        )

    def _recovery_section(self, recent_workouts, recent_activities):
        # Last night's sleep = sleep_hours from the most recent workout log (most recent = yesterday)
        last_sleep = None
        sleep_history = []
        for w in recent_workouts:
            if w.get("sleep_hours") is not None:
                if last_sleep is None:
                    last_sleep = w["sleep_hours"]
                sleep_history.append(w["sleep_hours"])

        sleep_lines = []
        if last_sleep is not None:
            sleep_lines.append(f"Last night's sleep: {last_sleep} hours")
        if sleep_history:
            avg = sum(sleep_history) / len(sleep_history)
            sleep_lines.append(
                f"Sleep over last {len(sleep_history)} nights: "
                f"{', '.join(f'{h:g}' for h in sleep_history)} (avg {avg:.1f})"
            )
        if not sleep_lines:
            sleep_lines.append("Sleep data: not yet logged")

        # Recent gym training
        if recent_workouts:
            gym_lines = ["Recent gym training (last days):"]
            for w in recent_workouts:
                gym_lines.append("- " + self._format_workout_log(w))
        else:
            gym_lines = ["Recent gym training: none logged"]

        # Recent athletic activity
        if recent_activities:
            ath_lines = ["Recent athletic activity (last days):"]
            for a in recent_activities:
                ath_lines.append("- " + self._format_activity_log(a))
        else:
            ath_lines = ["Recent athletic activity: none logged"]

        return (
            "=== RECOVERY STATE ===\n"
            + "\n".join(sleep_lines) + "\n\n"
            + "\n".join(gym_lines) + "\n\n"
            + "\n".join(ath_lines)
        )

    def _weather_section(self, weather, user_doc):
        if not weather:
            return "=== WEATHER ===\nweather data unavailable"
        # Weather is mostly relevant for outdoor athletes/endurance
        goals = (user_doc.get("profile") or {}).get("fitness_goals") or []
        outdoor_relevant = any(g in goals for g in ("endurance", "sports_training"))
        suffix = "" if outdoor_relevant else " - largely irrelevant for this athlete (gym-based)"
        return f"=== WEATHER ===\n{weather.get('summary', 'unavailable')}{suffix}"

    # ----- Task sections (the big difference between call types) -----

    def _daily_task_section(self, target_date):
        date_str = target_date.strftime('%A, %B %d')
        return (
            "=== YOUR TASK ===\n"
            f"Recommend {date_str}'s training session. Output ONLY this JSON, no markdown, no commentary:\n"
            "{\n"
            '  "is_rest_day": boolean,\n'
            '  "recommended_time": "HH:MM" or null,\n'
            '  "workout_type": "Concise label only - 8 WORDS MAXIMUM. e.g. \\"Pull-focused upper, hypertrophy\\" or \\"Zone 2 Cardio\\". DO NOT list exercises/sets/reps here." or null,\n'
            '  "intensity": "low" | "medium" | "high" or null,\n'
            '  "duration_minutes": integer or null,\n'
            '  "reasoning": "3-4 sentences. Reference the SPECIFIC principles you applied (which research, '
            'which constraint, which recovery state). Why THIS session for THIS athlete on THIS day, not '
            'generic coaching. Specific exercise/set/rep prescriptions may go here if relevant."\n'
            "}"
        )

    def _weekly_constraints_section(self, user_doc, semester_doc, week_start):
        """For the weekly call we list each day's commitments since the prompt covers 7 days."""
        lines = ["=== WEEK COMMITMENTS (per day) ==="]
        current_week = user_doc.get("current_week") or {}
        for offset in range(7):
            day = week_start + timedelta(days=offset)
            dow = self._day_of_week(day)
            classes = self._format_classes(semester_doc, dow)
            work = self._format_work(current_week.get("work_blocks", []), dow)
            study = self._format_study(current_week.get("study_hours"), dow)
            lines.append(
                f"{day.strftime('%a %m/%d')} ({dow}): "
                f"classes [{classes}], work [{work}], study [{study}]"
            )
        return "\n".join(lines)

    def _weekly_task_section(self, week_start, full_week_start=None):
        """Build the task section. If week_start > full_week_start, we're regenerating
        a partial week (mid-week reset based on logs)."""
        if full_week_start is None:
            full_week_start = week_start
        is_partial = week_start.date() > full_week_start.date()
        sunday = full_week_start + timedelta(days=6)
        num_days = (sunday.date() - week_start.date()).days + 1

        if is_partial:
            scope_line = (
                f"Plan {num_days} days of training, starting {week_start.strftime('%A %B %d')} "
                f"through {sunday.strftime('%A %B %d')}. The earlier days of this week are LOCKED "
                "(see 'PAST DAYS THIS WEEK' above) — do not output recommendations for them. "
                "Use the past days as historical context to inform what comes next."
            )
        else:
            scope_line = (
                f"Plan this athlete's full training week starting {week_start.strftime('%A %B %d')} "
                "(Monday through Sunday)."
            )

        return (
            "=== YOUR TASK ===\n"
            f"{scope_line} Distribute volume, intensity, and rest days to maximize adaptation while respecting:\n"
            "- Their commitments each day (classes, work, athletic practices, games)\n"
            "- Recovery between hard sessions (48-72 hours for same muscle group)\n"
            "- Their sleep pattern and energy distribution\n"
            "- Their injuries\n"
            "- The hard rules above\n"
            f"- {'WHAT THE ATHLETE ACTUALLY DID this week, not the original plan' if is_partial else 'A balanced full-week structure'}\n\n"
            "Output ONLY this JSON, no markdown, no commentary:\n"
            "{\n"
            '  "week_summary": "2-3 sentences explaining the structure ' +
            ('of the remainder of the week and how you adjusted from what was actually done' if is_partial
             else 'of this week and why') + '",\n'
            '  "days": [\n'
            "    {\n"
            '      "date": "YYYY-MM-DD",\n'
            '      "is_rest_day": boolean,\n'
            '      "recommended_time": "HH:MM" or null,\n'
            '      "workout_type": "Concise label only - 8 WORDS MAXIMUM. e.g. \\"Hypertrophy - Push (Chest/Shoulders/Triceps)\\" or \\"Zone 2 Cardio\\". DO NOT include exercise lists, sets, or reps here." or null,\n'
            '      "intensity": "low" | "medium" | "high" or null,\n'
            '      "duration_minutes": integer or null,\n'
            '      "reasoning": "1-2 sentences specific to this day\'s role in the week. Specific exercises/sets/reps may go here if relevant."\n'
            "    }\n"
            f"    // {num_days} entries total, dates from {week_start.strftime('%Y-%m-%d')} to {sunday.strftime('%Y-%m-%d')}\n"
            "  ]\n"
            "}"
        )

    def _original_plan_section(self, original):
        if not original:
            return "=== ORIGINAL PLAN FOR TODAY ===\nNone (no weekly plan was generated)"
        bits = []
        if original.get("is_rest_day"):
            bits.append("rest day")
        else:
            if original.get("workout_type"):    bits.append(original["workout_type"])
            if original.get("intensity"):       bits.append(f"intensity {original['intensity']}")
            if original.get("duration_minutes"):bits.append(f"{original['duration_minutes']} min")
            if original.get("recommended_time"):bits.append(f"at {original['recommended_time']}")
        return (
            "=== ORIGINAL PLAN FOR TODAY (from Sunday) ===\n"
            + ", ".join(bits) + "\n"
            + f"Reasoning: {original.get('reasoning', '(none recorded)')}"
        )

    def _daily_revision_task_section(self, target_date):
        return (
            "=== YOUR TASK ===\n"
            f"Revise {target_date.strftime('%A')}'s recommendation given the deviation above. "
            "Output the same daily JSON shape, with reasoning that explicitly references what you "
            "adjusted and why:\n"
            "{\n"
            '  "is_rest_day": boolean,\n'
            '  "recommended_time": "HH:MM" or null,\n'
            '  "workout_type": "Concise label - 8 WORDS MAXIMUM. No exercise lists." or null,\n'
            '  "intensity": "low" | "medium" | "high" or null,\n'
            '  "duration_minutes": integer or null,\n'
            '  "reasoning": "3-4 sentences. State what changed, which rule applied, and how you adjusted."\n'
            "}"
        )

    # =================================================================
    # Field formatters
    # =================================================================

    @staticmethod
    def _day_of_week(dt):
        return DAYS[dt.weekday()]

    @staticmethod
    def _goals_phrase(goals):
        if not goals: return "general fitness"
        return " + ".join(goals)

    @staticmethod
    def _budget_phrase(usd):
        if usd is None:
            return "unknown gym access"
        if usd == 0:
            return "no gym budget (bodyweight + outdoor only)"
        if usd <= 15:
            return f"${usd}/month - likely campus gym only (basic equipment)"
        if usd <= 40:
            return f"${usd}/month - budget chain gym (Planet Fitness tier; full machines + free weights)"
        if usd <= 80:
            return f"${usd}/month - mid-tier gym (full free weights, machines, cardio equipment)"
        return f"${usd}/month - premium gym, full equipment + classes"

    @staticmethod
    def _energy_phrase(levels):
        if not levels:
            return "unknown"
        m, a, e = levels.get('morning', 5), levels.get('afternoon', 5), levels.get('evening', 5)
        peak = max(m, a, e)
        if m == peak and m > a and m > e:
            chrono = " (clear morning chronotype)"
        elif e == peak and e > m and e > a:
            chrono = " (clear evening chronotype)"
        else:
            chrono = ""
        return f"morning {m}/10, afternoon {a}/10, evening {e}/10{chrono}"

    @staticmethod
    def _format_classes(semester_doc, dow):
        if not semester_doc:
            return "no semester data on file"
        today = [c for c in semester_doc.get("classes", []) if c["day_of_week"] == dow]
        today.sort(key=lambda c: c["start_time"])
        if not today:
            return "none"
        return "; ".join(f"{c['label']} ({c['start_time']}-{c['end_time']})" for c in today)

    @staticmethod
    def _format_work(work_blocks, dow):
        today = [b for b in work_blocks if b["day_of_week"] == dow]
        today.sort(key=lambda b: b["start_time"])
        if not today:
            return "none"
        return "; ".join(f"{b['start_time']}-{b['end_time']}" for b in today)

    @staticmethod
    def _format_study(study_hours, dow):
        if not study_hours:
            return "none planned"
        h = study_hours.get(dow, 0)
        if h == 0:
            return "none planned"
        if h >= 4:
            return f"{h} hours (heavy day)"
        return f"{h} hours"

    @staticmethod
    def _format_workout_log(w):
        date = w["date"].strftime("%a %b %d")
        sleep_note = f" [slept {w['sleep_hours']:g}hrs]" if w.get('sleep_hours') is not None else ""
        if not w["did_workout"]:
            return f"{date}: rest day{sleep_note}"
        bits = [date, w.get("workout_type") or "(no type)"]
        if w.get("duration_minutes"):
            bits.append(f"{w['duration_minutes']} min")
        if w.get("intensity"):
            bits.append(f"intensity {w['intensity']}/10")
        result = ", ".join(bits) + sleep_note
        if w.get("notes"):
            result += f" — note: {w['notes']}"
        return result

    @staticmethod
    def _format_activity_log(a):
        date = a["date"].strftime("%a %b %d")
        return (
            f"{date}: {a['activity_type']}, {a['duration_minutes']} min, "
            f"intensity {a['intensity']}/10 - {a['description']}"
        )

    @staticmethod
    def _available_windows(semester_doc, current_week, todays_activities, dow):
        """Compute free time blocks today by inverting class/work/activity blocks.
        Hours covered: 06:00 to 22:00 (training-relevant range)."""
        busy = []  # list of (start_minute, end_minute) tuples

        def to_min(s):
            h, m = s.split(":"); return int(h) * 60 + int(m)

        def from_min(m):
            return f"{m // 60:02d}:{m % 60:02d}"

        # Classes
        if semester_doc:
            for c in semester_doc.get("classes", []):
                if c["day_of_week"] == dow:
                    busy.append((to_min(c["start_time"]), to_min(c["end_time"])))
        # Work
        for b in (current_week or {}).get("work_blocks", []):
            if b["day_of_week"] == dow:
                busy.append((to_min(b["start_time"]), to_min(b["end_time"])))
        # Athletic activities — assume 2 hours surrounding for warmup/cooldown won't help; treat as logged duration only
        # We don't have start times for activities (only durations), so we omit them from window math.

        if not busy:
            return "06:00-22:00 (no commitments)"

        busy.sort()
        # Merge overlapping
        merged = [busy[0]]
        for s, e in busy[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Invert against 06:00-22:00
        DAY_START, DAY_END = 6 * 60, 22 * 60
        windows = []
        cursor = DAY_START
        for s, e in merged:
            if s > cursor:
                windows.append((cursor, min(s, DAY_END)))
            cursor = max(cursor, e)
            if cursor >= DAY_END:
                break
        if cursor < DAY_END:
            windows.append((cursor, DAY_END))

        # Drop windows shorter than 45 minutes (not useful for training)
        windows = [w for w in windows if w[1] - w[0] >= 45]
        if not windows:
            return "no usable training window today"
        return ", ".join(f"{from_min(s)}-{from_min(e)}" for s, e in windows)


# =====================================================================
# 10. Self-test
# =====================================================================

if __name__ == "__main__":
    print(f"Connected to MongoDB: {MONGO_DB_NAME}")
    print(f"Collections: {db.list_collection_names()}\n")

    # Synthetic athlete to demo the new prompt
    fake_user = {
        "username": "test",
        "profile": {
            "fitness_goals": ["hypertrophy", "sports_training"],
            "sport": "basketball",
            "team_context": "school",
            "budget_monthly_usd": 50,
            "energy_levels": {"morning": 9, "afternoon": 7, "evening": 7},
            "injuries_limitations": "mild left knee tendinitis, avoid deep loaded knee flexion",
            "location": {"city": "Corona, CA"},
        },
        "current_week": {
            "work_blocks": [
                {"day_of_week": "mon", "start_time": "16:00", "end_time": "20:00"},
            ],
            "study_hours": {"mon": 4.0, "tue": 2.0, "wed": 3.0, "thu": 2.0,
                            "fri": 1.0, "sat": 0.0, "sun": 2.0},
        },
    }
    fake_semester = {
        "classes": [
            {"label": "6090", "day_of_week": "mon", "start_time": "08:00", "end_time": "10:00"},
            {"label": "4080", "day_of_week": "mon", "start_time": "14:00", "end_time": "16:00"},
        ],
    }
    fake_workouts = [
        {"date": datetime(2026, 4, 26), "did_workout": True, "sleep_hours": 7.5,
         "workout_type": "chest+back lifting", "duration_minutes": 80, "intensity": 8},
        {"date": datetime(2026, 4, 25), "did_workout": False, "sleep_hours": 6.0},
        {"date": datetime(2026, 4, 24), "did_workout": True, "sleep_hours": 8.0,
         "workout_type": "legs", "duration_minutes": 60, "intensity": 7,
         "notes": "knees flagged tendinitis afterward"},
        {"date": datetime(2026, 4, 23), "did_workout": True, "sleep_hours": 7.0,
         "workout_type": "upper push", "duration_minutes": 50, "intensity": 6},
    ]
    fake_activities = [
        {"date": datetime(2026, 4, 25), "activity_type": "practice",
         "duration_minutes": 90, "intensity": 7,
         "description": "full-court drills + scrimmage"},
        {"date": datetime(2026, 4, 23), "activity_type": "practice",
         "duration_minutes": 75, "intensity": 6,
         "description": "shooting + half-court"},
    ]
    fake_weather = {"summary": "68F partly cloudy, no precipitation"}

    pb = PromptBuilder()
    sys_p, user_p = pb.build_daily(
        fake_user, fake_semester, fake_workouts, fake_activities, fake_weather,
        target_date=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )
    print("===== SYSTEM =====\n" + sys_p)
    print("\n===== USER =====\n" + user_p)