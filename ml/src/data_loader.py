"""
Data Ingestion and Integrity Module for Pima Indians Diabetes Dataset.
Handles automated fetching, validation, and schema verification.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

DATA_URLS = [
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
]

EXPECTED_COLUMNS = [
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

DEFAULT_RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "diabetes.csv"


def download_dataset(target_path: Path = DEFAULT_RAW_PATH) -> Path:
    """
    Downloads the authentic 768-record Pima Indians Diabetes dataset
    if not present locally.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.stat().st_size > 5000:
        return target_path

    for url in DATA_URLS:
        try:
            print(f"[DataLoader] Attempting to download dataset from: {url}")
            urllib.request.urlretrieve(url, target_path)

            # Check if brownlee raw format without headers
            df = pd.read_csv(target_path)
            if df.columns[0] != "Pregnancies":
                if df.shape[1] == 9:
                    df.columns = EXPECTED_COLUMNS
                    df.to_csv(target_path, index=False)
            
            if df.shape[0] in (767, 768) and df.shape[1] == 9:
                print(f"[DataLoader] Successfully verified and saved dataset to {target_path}")
                return target_path
        except Exception as e:
            print(f"[DataLoader] Download from {url} failed: {e}. Trying fallback...")

    raise RuntimeError(
        "Could not automatically download the Pima Indians dataset. "
        "Please ensure internet connectivity or manually place diabetes.csv in data/raw/"
    )


def load_raw_data(data_path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Loads and validates the Pima Indians Diabetes dataset.
    
    Returns:
        pd.DataFrame: 768 rows x 9 columns DataFrame.
    """
    data_path = Path(data_path)
    if not data_path.exists() or data_path.stat().st_size < 5000:
        download_dataset(data_path)

    df = pd.read_csv(data_path)

    # Validate column integrity
    if list(df.columns) != EXPECTED_COLUMNS:
        if df.shape[1] == 9:
            df.columns = EXPECTED_COLUMNS
            df.to_csv(data_path, index=False)
        else:
            raise ValueError(
                f"Dataset columns mismatch. Expected: {EXPECTED_COLUMNS}, got: {list(df.columns)}"
            )

    if df.shape[0] != 768:
        print(f"[DataLoader Warning] Expected 768 records, found {df.shape[0]} records.")

    return df


def get_features_and_target(
    data_path: Path = DEFAULT_RAW_PATH
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns predictor features X and target outcome y.
    """
    df = load_raw_data(data_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)
    return X, y


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Data successfully loaded. Shape: {df.shape}")
    print(df.head())
    print("\nClass distribution:\n", df["Outcome"].value_counts(normalize=True))
