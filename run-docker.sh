#!/usr/bin/env bash
# run-docker.sh — Build and start backend via Docker Compose
set -e

cd "$(dirname "$0")"

echo "────────────────────────────────────────────"
echo " RoadVisionAI — Docker backend"
echo "────────────────────────────────────────────"

if ! command -v docker &>/dev/null; then
  echo "❌  docker not found. Install Docker Desktop."
  exit 1
fi

mkdir -p uploads models

echo "→  Building image (first time takes ~60s)..."
docker compose build

echo ""
echo "→  Starting backend on http://localhost:8000"
echo "→  Open frontend/index.html in your browser"
echo "→  Logs below (Ctrl+C to stop):"
echo ""

docker compose up