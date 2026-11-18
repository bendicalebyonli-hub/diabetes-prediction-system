#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "  Pima Indians Diabetes Prediction System - Automated Launcher"
echo "======================================================================"
echo ""

echo "[1/5] Installing Python Dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/5] Running Clinical EDA & Generating Visualization Plots..."
python -m ml.src.eda

echo ""
echo "[3/5] Running 5-Fold Stratified CV Benchmarking & Pipeline Training..."
python -m ml.src.train

echo ""
echo "[4/5] Executing Automated Pytest Suite..."
pytest backend/tests/ -v

echo ""
echo "[5/5] Starting FastAPI Backend and Next.js Frontend..."
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!

echo ""
echo "======================================================================"
echo "  All services launched!"
echo "  - Frontend: http://localhost:3000"
echo "  - FastAPI Docs: http://127.0.0.1:8000/docs"
echo "======================================================================"

wait $BACKEND_PID $FRONTEND_PID
