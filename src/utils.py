# src/utils.py

import os
import joblib
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────

def load_all_models():
    """Load all serialized models and artifacts from the models/ directory."""
    paths = {
        "decision_tree":     os.path.join(BASE, "models", "decision_tree.pkl"),
        "kmeans":            os.path.join(BASE, "models", "kmeans.pkl"),
        "linear_regression": os.path.join(BASE, "models", "linear_regression.pkl"),
        "scaler":            os.path.join(BASE, "models", "scaler.pkl"),
        "label_encoder":     os.path.join(BASE, "models", "label_encoder.pkl"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            f"Run models.py first to train and serialize all models."
        )
    return {k: joblib.load(p) for k, p in paths.items()}


# ─────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────

FIELD_BOUNDS = {
    "N":           (0,    300,  "Nitrogen must be between 0 and 300 kg/ha."),
    "P":           (0,    150,  "Phosphorus must be between 0 and 150 kg/ha."),
    "K":           (0,    210,  "Potassium must be between 0 and 210 kg/ha."),
    "temperature": (0,    50,   "Temperature must be between 0 and 50 °C."),
    "humidity":    (0,    100,  "Humidity must be between 0 and 100%."),
    "ph":          (0,    14,   "Soil pH must be between 0 and 14."),
    "rainfall":    (0,    1000, "Rainfall must be between 0 and 1000 mm."),
}

def validate_inputs(values: dict) -> list[str]:
    """
    Validate input values against agronomic bounds.
    Returns a list of error messages (empty if all valid).
    """
    errors = []
    for key, value in values.items():
        lo, hi, msg = FIELD_BOUNDS[key]
        if not (lo <= value <= hi):
            errors.append(msg)
    return errors


# ─────────────────────────────────────────
# FEATURE PREPARATION
# ─────────────────────────────────────────

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def prepare_input(values: dict, scaler) -> np.ndarray:
    """
    Convert a dict of field values to a scaled numpy array
    ready for model inference.
    """
    arr = np.array([values[k] for k in FEATURE_ORDER]).reshape(1, -1)
    return scaler.transform(arr)


# ─────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────

def predict_crop(arr_scaled, dt_model, label_enc) -> str:
    """Run Decision Tree and return crop name string."""
    encoded = dt_model.predict(arr_scaled)[0]
    return label_enc.inverse_transform([encoded])[0]


def predict_cluster(arr_scaled, km_model) -> int:
    """Run KMeans and return cluster index."""
    return int(km_model.predict(arr_scaled)[0])


def predict_yield(arr_scaled, lr_model) -> tuple[float, float, float]:
    """
    Run Linear Regression.
    Returns (predicted_yield, lower_bound, upper_bound)
    with ±8% confidence margin.
    """
    yld    = float(lr_model.predict(arr_scaled)[0])
    margin = yld * 0.08
    return yld, yld - margin, yld + margin


# ─────────────────────────────────────────
# AGRONOMIC GUIDANCE
# ─────────────────────────────────────────

CLUSTER_GUIDANCE = {
    0: "Low-to-moderate nutrient zone. Recommend legume rotation and organic compost before planting.",
    1: "High-fertility zone. Ideal for nutrient-demanding crops such as rice, maize, or banana.",
    2: "Balanced soil profile. Suitable for most crops with standard fertilisation schedules.",
    3: "Saline or degraded zone. Consider salt-tolerant varieties and improved drainage systems.",
}

def get_cluster_guidance(cluster_id: int) -> str:
    return CLUSTER_GUIDANCE.get(
        cluster_id,
        f"Cluster {cluster_id} - Consult an agronomist for a detailed soil profile assessment."
    )


CROP_EMOJI = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫛",
    "coconut": "🥥", "coffee": "☕", "cotton": "🌿", "grapes": "🍇",
    "jute": "🌾", "kidneybeans": "🫘", "lentil": "🫛", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🫘", "mungbean": "🫘", "muskmelon": "🍈",
    "orange": "🍊", "papaya": "🍈", "pigeonpeas": "🫛", "pomegranate": "🍎",
    "rice": "🌾", "watermelon": "🍉",
}

def get_crop_emoji(crop_name: str) -> str:
    return CROP_EMOJI.get(crop_name.lower(), "🌱")


# ─────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────

def format_yield(yld: float, lower: float, upper: float) -> tuple[str, str]:
    """Returns (main display string, confidence bound string)."""
    main = f"{yld:,.0f} hg/ha"
    bounds = f"Confidence bound ±8%:  {lower:,.0f} – {upper:,.0f} hg/ha"
    return main, bounds


def format_crop(crop_name: str) -> str:
    return f"{get_crop_emoji(crop_name)}  {crop_name.capitalize()}"


# ─────────────────────────────────────────
# RESULTS EXPORT
# ─────────────────────────────────────────

def save_prediction_log(values: dict, crop: str, cluster: int, yld: float):
    """
    Append a prediction result to results/prediction_log.csv.
    Useful for demo and documentation purposes.
    """
    import csv
    from datetime import datetime

    log_path = os.path.join(BASE, "results", "prediction_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_exists = os.path.exists(log_path)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **values,
        "predicted_crop":    crop,
        "soil_cluster":      cluster,
        "predicted_yield":   round(yld, 2),
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Loading models...")
    models = load_all_models()

    sample = {"N": 90, "P": 42, "K": 43,
              "temperature": 20.8, "humidity": 82.0,
              "ph": 6.5, "rainfall": 202.9}

    errors = validate_inputs(sample)
    if errors:
        print("Validation errors:", errors)
    else:
        arr = prepare_input(sample, models["scaler"])
        crop    = predict_crop(arr,    models["decision_tree"], models["label_encoder"])
        cluster = predict_cluster(arr, models["kmeans"])
        yld, lo, hi = predict_yield(arr, models["linear_regression"])

        print(f"Crop:    {format_crop(crop)}")
        print(f"Cluster: {cluster} - {get_cluster_guidance(cluster)}")
        print(f"Yield:   {format_yield(yld, lo, hi)[0]}")
        print(f"Bounds:  {format_yield(yld, lo, hi)[1]}")
        save_prediction_log(sample, crop, cluster, yld)
        print("Log saved to results/prediction_log.csv")