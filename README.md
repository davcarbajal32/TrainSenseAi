# TrainSense

LLM-driven workout recommendations for college students. Considers academic schedule, fitness goals, budget, energy patterns, and workout history to suggest optimal workout times and types daily.

## Status

**Session 1:** Foundation — Flask + MongoDB + auth.

## Local development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then fill in SECRET_KEY and MONGO_URI
python app.py
```

Visit http://localhost:5050

Generate a SECRET_KEY:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Deploying to EC2

Your existing `run.sh` and GitHub Actions workflow already handle this. The new app is a drop-in replacement.

**One-time EC2 setup** (only needed because the app now reads from `.env`):

```bash
ssh into your EC2 box
cd ~/TrainSenseAi
nano .env       # paste in SECRET_KEY and MONGO_URI from your local .env
chmod 600 .env  # only you can read it
```

After that, `git push` to main triggers the GitHub Action which SSHes in, pulls, and restarts the app.

## Endpoints

| Method | Path             | Auth | Purpose                |
|--------|------------------|------|------------------------|
| GET    | /                | no   | landing page (HTML)    |
| GET    | /api/health      | no   | server + db status     |
| POST   | /api/auth/signup | no   | create user, log in    |
| POST   | /api/auth/login  | no   | start session          |
| POST   | /api/auth/logout | yes  | end session            |
| GET    | /api/auth/whoami | no   | check current session  |

## Collections

| Collection      | Purpose                                        |
|-----------------|------------------------------------------------|
| users           | identity + profile + current+past weekly plans |
| semesters       | one active semester per user with classes      |
| workouts        | history of daily logs (incl. rest days)        |
| recommendations | one Claude-generated rec per user per date     |

Document shapes are documented inline in `app.py`. MongoDB itself is schemaless — that's a reference, not enforcement.

## What's coming

- **Session 2:** profile setup, semester management, weekly plans, workout logging, prompt builder
- **Session 3:** Claude API integration, weather lookup, daily recommendation generation
- **Session 4:** polish, frontend templates for each page
