# Cadence AI

Cadence AI is an LLM-powered fitness planning app built for college students. It reads your class schedule, work shifts, study load, sleep, and recovery to generate a personalized weekly training plan — and adjusts whenever life gets in the way.

Built with Flask, MongoDB, and Claude AI. Deployed on AWS EC2.

Live at: https://cadenceai.mooo.com

## Project Structure

├── app.py              # Flask routes and API endpoints
├── main.py             # Core business logic, database models, prompt builder
├── claude.py           # Anthropic API client
├── weather.py          # Open-Meteo weather and geocoding
├── templates/
│   └── index.html      # Single-page frontend (Tailwind CSS)
├── requirements.txt
└── .env                # Secret keys (not committed)

## Stack

- **Backend:** Python, Flask, Flask-Login
- **Database:** MongoDB Atlas
- **AI:** Anthropic Claude Sonnet
- **Frontend:** Vanilla JS, Tailwind CSS
- **Deployment:** AWS EC2, nginx, GitHub Actions CI/CD
