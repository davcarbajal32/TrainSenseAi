#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/TrainSenseAi"
APP_FILE="app.py"
PY="$APP_DIR/.venv/bin/python"

cd "$APP_DIR"

echo "Pulling latest code..."
git fetch --all
git reset --hard origin/main

echo "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3.12 -m venv .venv
fi

echo "Installing dependencies..."
"$PY" -m pip install -U pip
"$PY" -m pip install -r requirements.txt

echo "Stopping old app..."
pkill -f "$PY $APP_FILE" || true

echo "Starting app..."
nohup "$PY" "$APP_FILE" > log.txt 2>&1 &

echo "Done. Check logs with:"
echo "tail -f log.txt"
