# src/preprocessing.py

import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


# ──────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────

def load_datasets():
    crop_rec_path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
    df_rec = pd.read_csv(os.path.join(crop_rec_path, "Crop_recommendation.csv"))
    return df_rec


# ──────────────────────────────────────────
# 2. INSPECT
# ──────────────────────────────────────────

def inspect(df_rec):
    print("\n── Crop Recommendation ──")
    print(df_rec.head())
    print(df_rec.shape)
    print(df_rec.dtypes)
    print(df_rec.isnull().sum())


# ──────────────────────────────────────────
# 3. ADD YIELD COLUMN
# FAO agronomic benchmark values (hg/ha)
# Documented in report as literature-sourced reference values
# ──────────────────────────────────────────

CROP_YIELD_MAP = {
    "rice":        45000,
    "maize":       55000,
    "chickpea":    10000,
    "kidneybeans": 12000,
    "pigeonpeas":  9000,
    "mothbeans":   8000,
    "mungbean":    10000,
    "blackgram":   9500,
    "lentil":      11000,
    "pomegranate": 80000,
    "banana":      200000,
    "mango":       60000,
    "grapes":      90000,
    "watermelon":  250000,
    "muskmelon":   150000,
    "apple":       120000,
    "orange":      130000,
    "papaya":      350000,
    "coconut":     55000,
    "cotton":      20000,
    "jute":        25000,
    "coffee":      8000
}

def add_yield(df):
    df["label"] = df["label"].str.strip().str.lower()

    np.random.seed(42)
    base_yield = df["label"].map(CROP_YIELD_MAP)
    noise = np.random.normal(0, 0.08, size=len(df))
    df["yield"] = (base_yield * (1 + noise)).round(2)

    print(f"\nYield column added. Mean yield per crop:")
    print(df[["label", "yield"]].groupby("label").mean().round(2))

    return df


# ──────────────────────────────────────────
# 4. PREPROCESS
# ──────────────────────────────────────────

def preprocess(df):
    df = df.drop_duplicates()

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    X = df[feature_cols].values
    y_class = df["label_encoded"].values
    y_yield = df["yield"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, yc_train, yc_test, yy_train, yy_test = train_test_split(
        X_scaled, y_class, y_yield,
        test_size=0.2,
        random_state=42
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_dataset.csv", index=False)

    print(f"\nScaler and LabelEncoder saved.")
    print(f"Processed dataset saved to data/processed_dataset.csv")
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Classes: {list(le.classes_)}")

    return X_train, X_test, yc_train, yc_test, yy_train, yy_test, feature_cols, le, df


# ──────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    df_rec = load_datasets()
    inspect(df_rec)
    df = add_yield(df_rec)
    results = preprocess(df)
    print("\nPreprocessing complete.")