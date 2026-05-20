# src/models.py

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    silhouette_score, mean_squared_error, mean_absolute_error, r2_score
)

from preprocessing import load_datasets, add_yield, preprocess

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ──────────────────────────────────────────
# 1. DECISION TREE - Crop Classification
# ──────────────────────────────────────────

def train_decision_tree(X_train, X_test, yc_train, yc_test, feature_cols, le):
    print("\n── Training Decision Tree Classifier ──")

    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    dt.fit(X_train, yc_train)
    y_pred = dt.predict(X_test)

    acc  = accuracy_score(yc_test, y_pred)
    prec = precision_score(yc_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(yc_test, y_pred, average="weighted", zero_division=0)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")

    # feature importance plot
    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sorted_features, sorted_importances, color="steelblue")
    ax.set_title("Decision Tree - Feature Importance")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()
    print("  Feature importance plot saved.")

    joblib.dump(dt, "models/decision_tree.pkl")
    print("  Model saved: models/decision_tree.pkl")

    return dt, {"accuracy": acc, "precision": prec, "recall": rec}


# ──────────────────────────────────────────
# 2. KMeans CLUSTERING - Soil Zone Segmentation
# ──────────────────────────────────────────

def train_kmeans(X_train, X_test):
    print("\n── Training KMeans Clustering ──")

    # find optimal k using silhouette score
    best_k, best_score, best_model = 2, -1, None
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_train)
        score = silhouette_score(X_train, labels)
        print(f"  k={k} -> silhouette: {score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    print(f"  Best k: {best_k} | Best silhouette: {best_score:.4f}")

    # cluster scatter plot (PC1 vs PC2 via simple projection)
    labels_all = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_test[:, 0], X_test[:, 1],
        c=labels_all, cmap="tab10", alpha=0.7, edgecolors="k", linewidths=0.3
    )
    ax.set_title(f"KMeans Clustering (k={best_k}) - Soil Zone Segmentation")
    ax.set_xlabel("Feature 1 (N - scaled)")
    ax.set_ylabel("Feature 2 (P - scaled)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig("results/cluster_scatter.png", dpi=150)
    plt.close()
    print("  Cluster scatter plot saved.")

    joblib.dump(best_model, "models/kmeans.pkl")
    print(f"  Model saved: models/kmeans.pkl")

    return best_model, {"best_k": best_k, "silhouette": best_score}


# ──────────────────────────────────────────
# 3. LINEAR REGRESSION - Yield Prediction
# ──────────────────────────────────────────

def train_linear_regression(X_train, X_test, yy_train, yy_test):
    print("\n── Training Linear Regression ──")

    lr = LinearRegression()
    lr.fit(X_train, yy_train)
    y_pred = lr.predict(X_test)

    rmse = np.sqrt(mean_squared_error(yy_test, y_pred))
    mae  = mean_absolute_error(yy_test, y_pred)
    r2   = r2_score(yy_test, y_pred)

    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE : {mae:.2f}")
    print(f"  R²  : {r2:.4f}")

    # residual plot
    residuals = yy_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="tomato", edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Linear Regression - Residual Plot")
    ax.set_xlabel("Predicted Yield (hg/ha)")
    ax.set_ylabel("Residual")
    plt.tight_layout()
    plt.savefig("results/residual_plot.png", dpi=150)
    plt.close()
    print("  Residual plot saved.")

    joblib.dump(lr, "models/linear_regression.pkl")
    print("  Model saved: models/linear_regression.pkl")

    return lr, {"rmse": rmse, "mae": mae, "r2": r2}


# ──────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    df_rec = load_datasets()
    df     = add_yield(df_rec)
    X_train, X_test, yc_train, yc_test, yy_train, yy_test, feature_cols, le, df = preprocess(df)

    dt,  dt_metrics  = train_decision_tree(X_train, X_test, yc_train, yc_test, feature_cols, le)
    km,  km_metrics  = train_kmeans(X_train, X_test)
    lr,  lr_metrics  = train_linear_regression(X_train, X_test, yy_train, yy_test)

    print("\n── All Models Trained ──")
    print(f"Decision Tree  -> Accuracy: {dt_metrics['accuracy']:.4f}")
    print(f"KMeans         -> Silhouette: {km_metrics['silhouette']:.4f} | k: {km_metrics['best_k']}")
    print(f"Linear Reg     -> R²: {lr_metrics['r2']:.4f} | RMSE: {lr_metrics['rmse']:.2f}")
    print("\nmodels.py complete.")