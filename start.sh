#!/usr/bin/env bash
# start.sh — Launch FinTrust Compass (API + UI)
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Activate venv
source venv/bin/activate

# Check ChromaDB exists
if [ ! -d "chroma_db" ]; then
  echo "[ERROR] ChromaDB not found. Run ingestion first:"
  echo "  python ingest.py --inspect"
  exit 1
fi

# Kill any existing processes on these ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

echo "Starting FinTrust Compass API on http://localhost:8000 ..."
python -W ignore -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 5
echo "Starting FinTrust Compass UI on http://localhost:8501 ..."
python -W ignore -m streamlit run frontend/app.py --server.port 8501 --server.headless true &
UI_PID=$!

echo ""
echo "============================================="
echo "  FinTrust Compass is running!"
echo "  API : http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo "  UI  : http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo "============================================="

trap "kill $API_PID $UI_PID 2>/dev/null; exit 0" INT TERM
wait
