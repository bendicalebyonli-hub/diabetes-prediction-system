import sys
from pathlib import Path

# Ensure root directory is on the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Export the FastAPI app for Vercel Serverless
from backend.app.main import app
