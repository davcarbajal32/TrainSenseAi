"""
TrainSense - Flask web layer.

Thin wrapper around main.py. Just HTTP plumbing.
"""

from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, render_template
from flask_login import (
    LoginManager,
    login_user, logout_user, login_required, current_user,
)

import main
from main import (
    db,
    User,
    PromptBuilder,
    SECRET_KEY,
    validate_signup, validate_login,
    validate_profile, validate_semester,
    validate_weekly_plan, validate_workout,
    validate_athletic_activity,
    create_semester, get_active_semester, deactivate_past_semesters,
    save_weekly_plan,
    log_workout, get_recent_workouts,
    log_athletic_activity, get_recent_athletic_activities,
    generate_weekly_recommendations,
    revise_today_recommendation,
    detect_deviation,
)
from bson import ObjectId
main.ObjectId = ObjectId  # so the recommendations endpoint can reach it via main


# =====================================================================
# Flask app
# =====================================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

login_manager = LoginManager()
login_manager.init_app(app)
prompt_builder = PromptBuilder()


@login_manager.user_loader
def load_user(user_id):
    return User.find_by_id(user_id)


@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "authentication required"}), 401


# =====================================================================
# Single page route - SPA handles client-side navigation
# =====================================================================

@app.get("/")
def home():
    return render_template("first.html")


# =====================================================================
# API: Health
# =====================================================================

@app.get("/api/health")
def health():
    try:
        db.command("ping")
        return jsonify({"status": "ok", "db": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "degraded", "error": str(e)}), 503


# =====================================================================
# API: Auth
# =====================================================================

@app.post("/api/auth/signup")
def signup():
    data = request.get_json(silent=True) or {}
    errors = validate_signup(data)
    if errors:
        return jsonify({"error": "validation failed", "details": errors}), 400

    user, err = User.create(data["username"], data["email"], data["password"])
    if err:
        return jsonify({"error": err}), 409

    login_user(user)
    return jsonify({
        "id": user.id, "username": user.username,
        "email": user.email, "profile_completed": user.profile_completed,
    }), 201


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    errors = validate_login(data)
    if errors:
        return jsonify({"error": "validation failed", "details": errors}), 400

    user, err = User.authenticate(data["email"], data["password"])
    if err:
        return jsonify({"error": err}), 401

    login_user(user)
    # Housekeeping: clean up any semesters whose end_date has passed.
    deactivate_past_semesters(user.id)
    return jsonify({
        "id": user.id, "username": user.username,
        "email": user.email, "profile_completed": user.profile_completed,
    }), 200


@app.post("/api/auth/logout")
@login_required
def logout():
    logout_user()
    return jsonify({"ok": True}), 200


@app.get("/api/auth/whoami")
def whoami():
    if not current_user.is_authenticated:
        return jsonify({"authenticated": False}), 200
    is_athlete = "sports_training" in (current_user.profile.get("fitness_goals") or [])
    return jsonify({
        "authenticated": True,
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "profile_completed": current_user.profile_completed,
        "is_athlete": is_athlete,
    }), 200


# =====================================================================
# API: Profile
# =====================================================================

@app.get("/api/profile")
@login_required
def get_profile():
    return jsonify({
        "profile_completed": current_user.profile_completed,
        "profile": current_user.profile,
    }), 200


@app.post("/api/profile")
@login_required
def update_profile():
    data = request.get_json(silent=True) or {}
    errors = validate_profile(data)
    if errors:
        return jsonify({"error": "validation failed", "details": errors}), 400

    current_user.update_profile(data)
    return jsonify({
        "ok": True,
        "profile_completed": current_user.profile_completed,
        "profile": current_user.profile,
    }), 200


# =====================================================================
# API: Semesters
# =====================================================================

@app.get("/api/semesters/active")
@login_required
def get_active_semester_api():
    sem = get_active_semester(current_user.id)
    if not sem:
        return jsonify({"semester": None}), 200
    sem["_id"] = str(sem["_id"])
    sem["user_id"] = str(sem["user_id"])
    sem["start_date"] = sem["start_date"].isoformat()
    sem["end_date"] = sem["end_date"].isoformat()
    sem["created_at"] = sem["created_at"].isoformat()
    return jsonify({"semester": sem}), 200


@app.post("/api/semesters")
@login_required
def create_semester_api():
    data = request.get_json(silent=True) or {}
    errors = validate_semester(data)
    if errors:
        return jsonify({"error": "validation failed", "details": errors}), 400

    sem = create_semester(
        user_id=current_user.id,
        label=data["label"],
        start_date=data["start_date"],
        end_date=data["end_date"],
        classes=data["classes"],
    )
    return jsonify({"ok": True, "semester_id": str(sem["_id"])}), 201


# =====================================================================
# API: Weekly plan
# =====================================================================

@app.get("/api/weekly-plan")
@login_required
def get_weekly_plan():
    user_doc = db.users.find_one({"username": current_user.username})
    plan = user_doc.get("current_week")
    if plan and plan.get("week_start"):
        plan = dict(plan)
        plan["week_start"] = plan["week_start"].isoformat()
    return jsonify({"current_week": plan}), 200


@app.post("/api/weekly-plan")
@login_required
def save_weekly_plan_api():
    data = request.get_json(silent=True) or {}
    errors = validate_weekly_plan(data)
    if errors:
        return jsonify({"error": "validation failed", "details": errors}), 400

    save_weekly_plan(
        user_id=current_user.id,
        week_start=data["week_start"],
        work_blocks=data["work_blocks"],
        study_hours=data["study_hours"],
    )
    return jsonify({"ok": True}), 200


# =====================================================================
# API: Workouts (with sleep)
# =====================================================================

@app.get("/api/workouts")
@login_required
def list_workouts():
    recent = get_recent_workouts(current_user.id, limit=10)
    for w in recent:
        w["_id"] = str(w["_id"])
        w["user_id"] = str(w["user_id"])
        w["date"] = w["date"].isoformat()
        w["logged_at"] = w["logged_at"].isoformat()
    return jsonify({"workouts": recent}), 200


@app.post("/api/workouts")
@login_required
def create_workout():
    data = request.get_json(silent=True) or {}
    errors = validate_workout(data)
    if errors:
        return jsonify({"error": "validation failed", "details": errors}), 400

    if "date" not in data:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        data["date"] = yesterday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    log_workout(
        user_id=current_user.id,
        date_in=data["date"],
        did_workout=data["did_workout"],
        sleep_hours=data.get("sleep_hours"),
        workout_type=data.get("workout_type"),
        duration_minutes=data.get("duration_minutes"),
        intensity=data.get("intensity"),
        notes=data.get("notes"),
        followed_plan=data.get("followed_plan"),
    )
    return jsonify({"ok": True}), 201


# =====================================================================
# API: Athletic activities (sports athletes only)
# =====================================================================

@app.get("/api/athletic-activities")
@login_required
def list_athletic_activities():
    recent = get_recent_athletic_activities(current_user.id, limit=20)
    for a in recent:
        a["_id"] = str(a["_id"])
        a["user_id"] = str(a["user_id"])
        a["date"] = a["date"].isoformat()
        a["logged_at"] = a["logged_at"].isoformat()
    return jsonify({"activities": recent}), 200


@app.post("/api/athletic-activities")
@login_required
def create_athletic_activity():
    data = request.get_json(silent=True) or {}
    errors = validate_athletic_activity(data)
    if errors:
        return jsonify({"error": "validation failed", "details": errors}), 400

    log_athletic_activity(
        user_id=current_user.id,
        date_in=data["date"],
        activity_type=data["activity_type"],
        duration_minutes=data["duration_minutes"],
        description=data["description"],
        intensity=data["intensity"],
    )
    return jsonify({"ok": True}), 201


# =====================================================================
# API: Recommendations (stub - real generation lands in session 3)
# =====================================================================

@app.get("/api/recommendations/week")
@login_required
def get_week_recommendations():
    """Returns recommendations for the current week (Mon-Sun).

    Each day returns a `state` field with one of three values:
      - 'followed':  user logged a check-in matching the plan's intent
                     (workout planned + did_workout=true + followed_plan=true,
                      OR rest planned + did_workout=false)
      - 'off_plan':  user logged a check-in that didn't match the plan
                     (workout planned + did_workout=true + followed_plan=false,
                      OR rest planned + did_workout=true)
      - 'empty':     no log, or planned workout + user took rest
                     (the user can't 'partially' complete a planned workout
                      by skipping it - that's just empty)
    """
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6, hours=23, minutes=59, seconds=59)

    recs = list(db.recommendations.find({
        "user_id": main.ObjectId(current_user.id),
        "date": {"$gte": monday, "$lte": sunday},
    }).sort("date", 1))

    workouts_this_week = list(db.workouts.find({
        "user_id": main.ObjectId(current_user.id),
        "date": {"$gte": monday, "$lte": sunday},
    }))
    logs_by_date = {w["date"].date(): w for w in workouts_this_week}

    completed_dates = set()  # state = 'followed'
    off_plan_dates  = set()  # state = 'off_plan'

    for r in recs:
        d = r["date"].date()
        log = logs_by_date.get(d)
        if not log:
            continue  # no log = empty

        rec_says_rest = bool(r.get("is_rest_day"))
        user_worked   = bool(log.get("did_workout"))
        followed_plan = log.get("followed_plan")  # nullable bool

        if rec_says_rest:
            # Plan said rest. Did they rest?
            if not user_worked:
                completed_dates.add(d)         # rested as planned
            else:
                off_plan_dates.add(d)          # worked out on a rest day
        else:
            # Plan said work out.
            if user_worked and followed_plan is True:
                completed_dates.add(d)         # did the planned workout
            elif user_worked and followed_plan is False:
                off_plan_dates.add(d)          # did a different workout
            elif user_worked and followed_plan is None:
                # Old log without the new field. Treat as off_plan rather
                # than green so we don't lie about completion.
                off_plan_dates.add(d)
            else:
                # User took rest on a planned workout day. Empty - they didn't
                # complete the plan, didn't work out either.
                pass

    out = []
    for r in recs:
        r["_id"] = str(r["_id"])
        r["user_id"] = str(r["user_id"])
        date_obj = r["date"]
        r["date"] = date_obj.isoformat()
        d = date_obj.date()
        if d in completed_dates:
            r["state"] = "followed"
        elif d in off_plan_dates:
            r["state"] = "off_plan"
        else:
            r["state"] = "empty"
        # Keep the old `completed` field for backward compat with old client code
        r["completed"] = (r["state"] == "followed")
        if r.get("generated_at"):
            r["generated_at"] = r["generated_at"].isoformat()
        out.append(r)

    return jsonify({
        "week_start": monday.isoformat(),
        "recommendations": out,
        "completed_dates": [d.isoformat() for d in completed_dates],
        "off_plan_dates":  [d.isoformat() for d in off_plan_dates],
    }), 200


@app.post("/api/recommendations/week")
@login_required
def generate_week_api():
    """Trigger Claude generation for the full week. Stores 7 daily docs."""
    if not current_user.profile_completed:
        return jsonify({"error": "Complete your profile before generating a plan"}), 400

    try:
        result = generate_weekly_recommendations(current_user.id)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    return jsonify({
        "ok": True,
        "week_summary": result["week_summary"],
        "days_stored": result["days_stored"],
    }), 201


@app.post("/api/recommendations/today/regenerate")
@login_required
def regenerate_today_api():
    """Regenerate today's recommendation. Optionally takes a deviation reason
    in the body; otherwise auto-detects from yesterday's check-in."""
    body = request.get_json(silent=True) or {}
    reason = body.get("reason")

    if not reason:
        # Auto-detect based on yesterday's logs
        detected = detect_deviation(current_user.id)
        reason = detected or "user requested fresh recommendation"

    try:
        doc = revise_today_recommendation(current_user.id, deviation_reason=reason)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    return jsonify({
        "ok": True,
        "reasoning": doc.get("reasoning"),
        "is_rest_day": doc.get("is_rest_day"),
    }), 201


# =====================================================================
# API: Prompt previews (debug - lets you see what would be sent to Claude)
# =====================================================================

@app.get("/api/prompt-preview")
@login_required
def prompt_preview_daily():
    user_doc, sem, recent_w, recent_a = prompt_builder.gather_context(current_user.id)
    weather = None  # session 3 fills this in
    sys_p, user_p = prompt_builder.build_daily(user_doc, sem, recent_w, recent_a, weather)
    return jsonify({"system_prompt": sys_p, "user_prompt": user_p}), 200


@app.get("/api/prompt-preview/weekly")
@login_required
def prompt_preview_weekly():
    user_doc, sem, recent_w, recent_a = prompt_builder.gather_context(current_user.id)
    weather = None
    # Use upcoming Monday as the week_start
    # Use naive UTC for consistency with Mongo's stored datetimes
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    days_until_monday = (0 - today.weekday()) % 7  # 0 = monday
    week_start = today + timedelta(days=days_until_monday or 7)  # next Monday (or this Monday if today)
    if today.weekday() == 0:  # if today IS Monday, plan from today
        week_start = today
    sys_p, user_p = prompt_builder.build_weekly(user_doc, sem, recent_w, recent_a, weather, week_start)
    return jsonify({"system_prompt": sys_p, "user_prompt": user_p}), 200


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)