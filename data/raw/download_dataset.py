"""
Standalone Dataset Downloader Utility for Pima Indians Diabetes Dataset.
Fetches the authentic 768-record dataset from authoritative repositories.
"""

import sys
from pathlib import Path
import urllib.request
import pandas as pd

MIRRORS = [
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
]

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

TARGET_PATH = Path(__file__).resolve().parent / "diabetes.csv"


def fetch_dataset():
    print(f"Target path: {TARGET_PATH}")
    if TARGET_PATH.exists() and TARGET_PATH.stat().st_size > 5000:
        df = pd.read_csv(TARGET_PATH)
        if len(df) in (767, 768):
            print(f"Authentic dataset already present with {len(df)} records.")
            return TARGET_PATH

    for url in MIRRORS:
        try:
            print(f"Attempting download from: {url}")
            urllib.request.urlretrieve(url, TARGET_PATH)
            df = pd.read_csv(TARGET_PATH)
            
            # If headerless format (like brownlee mirror)
            if df.columns[0] != "Pregnancies" and df.shape[1] == 9:
                df.columns = COLUMNS
                df.to_csv(TARGET_PATH, index=False)
            
            print(f"Successfully downloaded and verified dataset ({len(df)} rows, 9 columns).")
            return TARGET_PATH
        except Exception as e:
            print(f"Mirror failed ({e}), attempting next mirror...")

    print("Error: Could not retrieve dataset from mirrors.")
    return None


if __name__ == "__main__":
    fetch_dataset()
