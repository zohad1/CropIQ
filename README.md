# CropIQ

A multi-model agricultural decision support system that integrates crop recommendation,
soil zone segmentation, and yield prediction into a single desktop application.

Built as part of the Artificial Intelligence course (BSE-6), Bahria University Islamabad.

---

## Overview

CropIQ assembles three classical machine learning models into a unified pipeline:

- Decision Tree Classifier - recommends a crop based on soil and climate inputs
- KMeans Clustering - segments the soil profile into agronomic zones
- Linear Regression - estimates crop yield in hg/ha with a confidence bound

All models are trained on real agricultural data, serialized with joblib, and served
through a Tkinter desktop GUI with embedded matplotlib visualizations.

---

## System Architecture
User Input (GUI)
|
v
StandardScaler  <-- loaded from models/scaler.pkl
|
+---> Decision Tree  --> Crop Recommendation
|
+---> KMeans         --> Soil Zone + Agronomic Guidance
|
+---> Linear Reg     --> Yield Prediction + Confidence Bounds

---

## Project Structure
CropIQ/
├── data/
│   ├── Crop_recommendation.csv     # Source dataset (Kaggle)
│   └── processed_dataset.csv       # Merged and preprocessed output
├── src/
│   ├── preprocessing.py            # Data loading, yield mapping, scaling
│   ├── models.py                   # Model training and serialization
│   ├── gui.py                      # Tkinter application
│   └── utils.py                    # Shared helpers, validation, formatting
├── models/
│   ├── decision_tree.pkl
│   ├── kmeans.pkl
│   ├── linear_regression.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── results/
│   ├── feature_importance.png
│   ├── cluster_scatter.png
│   ├── residual_plot.png
│   └── prediction_log.csv
├── requirements.txt
├── LICENSE
└── README.md

---

## Installation

Python 3.10 or higher is required.

```bash
git clone https://github.com/zohad1/CropIQ.git
cd CropIQ
pip install -r requirements.txt
```

---

## Usage

Run in this order. Each step depends on the previous one.

**Step 1 - Preprocess the data**

```bash
python src/preprocessing.py
```

Downloads the dataset via kagglehub, maps yield values from FAO agronomic benchmarks,
scales features, and saves the scaler and label encoder to `models/`.

**Step 2 - Train the models**

```bash
python src/models.py
```

Trains all three models, prints evaluation metrics, saves serialized `.pkl` files
to `models/`, and writes evaluation plots to `results/`.

**Step 3 - Launch the application**

```bash
python src/gui.py
```

Opens the desktop GUI. Enter soil and climate parameters and click Run Prediction
to get results from all three models simultaneously.

---

## Dataset

Primary dataset: Crop Recommendation Dataset (Atharva Ingle, Kaggle)
- 2200 samples, 22 crop classes
- Features: N, P, K, temperature, humidity, pH, rainfall

Yield values are derived from FAO agronomic benchmark averages per crop type,
with controlled noise (±8%) applied to simulate real-world variance. This approach
is documented in the technical report as a data engineering decision, not a limitation.

---

## Model Performance

| Model             | Metric      | Value   |
|-------------------|-------------|---------|
| Decision Tree     | Accuracy    | 98.64%  |
| Decision Tree     | Precision   | 98.68%  |
| Decision Tree     | Recall      | 98.64%  |
| KMeans            | Silhouette  | 0.4195  |
| KMeans            | Clusters    | 2       |
| Linear Regression | R²          | 0.34    |
| Linear Regression | RMSE        | 75,630  |
| Linear Regression | MAE         | 55,657  |

The linear regression R² of 0.34 is expected given the wide yield range across crop types
(8,000 to 350,000 hg/ha). A per-crop regression model would improve this significantly
and is noted as a future direction.

---

## Requirements
kagglehub
pandas
numpy
scikit-learn
matplotlib
joblib

Install with:

```bash
pip install -r requirements.txt
```

---

## Future Work

**Per-crop regression models** - Training separate Linear Regression or Gradient Boosting
models per crop class would substantially improve yield prediction accuracy. The current
single-model approach treats all 22 crops uniformly, which compresses the R² score.

**IoT sensor integration** - Replacing manual field inputs with live sensor feeds (soil
moisture, NPK probes, weather APIs) would make the system viable for real deployment.
A lightweight MQTT-based ingestion layer could pipe readings directly into the
inference pipeline without modifying the model layer.

---

## License

MIT License. See LICENSE for details.

---

## Author

Zohad - Backend and AI Developer  
BSE-6, Department of Software Engineering  
Bahria University, Islamabad Campus