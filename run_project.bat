@echo off
echo ======================================================================
echo   Pima Indians Diabetes Prediction System - Automated Launcher
echo ======================================================================
echo.

echo [1/5] Installing Python Dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error installing Python requirements.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/5] Running Clinical EDA & Generating Visualization Plots...
python -m ml.src.eda

echo.
echo [3/5] Running 5-Fold Stratified CV Benchmarking & Pipeline Training...
python -m ml.src.train

echo.
echo [4/5] Executing Automated Pytest Suite...
pytest backend/tests/ -v

echo.
echo [5/5] Launching FastAPI Backend Service (Port 8000)...
start "FastAPI Backend" uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

echo.
echo Launching Next.js Frontend (Port 3000)...
cd frontend
if not exist node_modules (
    echo Installing npm dependencies...
    npm install
)
start "Next.js Frontend" npm run dev

echo.
echo ======================================================================
echo   All services launched!
echo   - Frontend: http://localhost:3000
echo   - FastAPI Docs: http://127.0.0.1:8000/docs
echo ======================================================================
pause
