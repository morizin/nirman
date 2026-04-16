#!/usr/bin/env bash
# run-native.sh — Start the backend natively on macOS (uses CoreML / MPS)
set -e

cd "$(dirname "$0")"

echo "────────────────────────────────────────────"
echo " RoadVisionAI — Native macOS backend"
echo "────────────────────────────────────────────"

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found. Install from https://python.org"
  exit 1
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "→  Creating virtual environment..."
  uv run python3 -m venv .venv
fi

source .venv/bin/activate

echo "→  Installing dependencies..."
uv add -r backend/requirements-native.txt

# Check onnxruntime providers
echo ""
echo "→  Checking ONNX providers..."
uv run python3 -c "
import onnxruntime as ort
p = ort.get_available_providers()
print('   Available:', p)
if 'CoreMLExecutionProvider' in p:
    print('   ✅  CoreML found — will use Apple GPU/ANE')
else:
    print('   ⚠️   CoreML not found — CPU inference only')
    print('       On Apple Silicon, install: pip install onnxruntime')
    print('       (the standard package includes CoreML EP on macOS 12+)')
"

echo ""
echo "→  Starting backend on http://localhost:8000"
echo "→  Open frontend/index.html in your browser"
echo "→  Press Ctrl+C to stop"
echo ""

python3 -m http.server 8001 &
cd backend
uv run python3 app.py
