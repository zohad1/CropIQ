# src/gui.py

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ──────────────────────────────────────────
# LOAD MODELS & ARTIFACTS
# ──────────────────────────────────────────

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_models():
    dt  = joblib.load(os.path.join(BASE, "models", "decision_tree.pkl"))
    km  = joblib.load(os.path.join(BASE, "models", "kmeans.pkl"))
    lr  = joblib.load(os.path.join(BASE, "models", "linear_regression.pkl"))
    sc  = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
    le  = joblib.load(os.path.join(BASE, "models", "label_encoder.pkl"))
    return dt, km, lr, sc, le

dt_model, km_model, lr_model, scaler, label_enc = load_models()

# agronomic guidance per cluster
CLUSTER_GUIDANCE = {
    0: "Cluster 0 - Low Nutrient Zone: Soil lacks key nutrients. Recommend legume rotation and organic compost application before planting.",
    1: "Cluster 1 - High Nutrient Zone: Fertile soil profile. Suitable for high-demand crops like maize, rice, or banana.",
    2: "Cluster 2 - Moderate Zone: Balanced nutrient levels. Suitable for most crops with standard fertilization.",
    3: "Cluster 3 - Saline Zone: High salt content detected. Recommend salt-tolerant crops and drainage improvement.",
}

def get_cluster_guidance(cluster_id):
    return CLUSTER_GUIDANCE.get(cluster_id, f"Cluster {cluster_id} - Consult agronomist for detailed soil analysis.")


# ──────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────

class CropIQApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CropIQ - Smart Agriculture Decision Support System")
        self.geometry("1100x750")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")

        self._build_header()
        self._build_notebook()

    # ── Header ──
    def _build_header(self):
        header = tk.Frame(self, bg="#2a9d8f", pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="🌾  CropIQ  |  Smart Agriculture Decision Support System",
            font=("Segoe UI", 16, "bold"),
            bg="#2a9d8f", fg="white"
        ).pack()
        tk.Label(
            header,
            text="Decision Tree  •  KMeans Clustering  •  Linear Regression",
            font=("Segoe UI", 9),
            bg="#2a9d8f", fg="#d4f1ef"
        ).pack()

    # ── Notebook (tabs) ──
    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2d2d44", foreground="white",
                        padding=[14, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "#2a9d8f")])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_predict = tk.Frame(nb, bg="#1e1e2e")
        self.tab_plots   = tk.Frame(nb, bg="#1e1e2e")

        nb.add(self.tab_predict, text="  Predict  ")
        nb.add(self.tab_plots,   text="  Visualizations  ")

        self._build_predict_tab()
        self._build_plots_tab()

    # ──────────────────────────────────────────
    # TAB 1 - PREDICT
    # ──────────────────────────────────────────

    def _build_predict_tab(self):
        tab = self.tab_predict

        # ── Input Panel ──
        input_frame = tk.LabelFrame(
            tab, text="  Soil & Climate Parameters  ",
            bg="#2d2d44", fg="#2a9d8f",
            font=("Segoe UI", 10, "bold"),
            padx=15, pady=15
        )
        input_frame.pack(side="left", fill="y", padx=(10, 5), pady=10)

        fields = [
            ("Nitrogen (N)",      "0",   "kg/ha"),
            ("Phosphorus (P)",    "0",   "kg/ha"),
            ("Potassium (K)",     "0",   "kg/ha"),
            ("Temperature",       "25",  "°C"),
            ("Humidity",          "70",  "%"),
            ("Soil pH",           "6.5", ""),
            ("Rainfall",          "200", "mm"),
        ]

        self.entries = {}
        for i, (label, default, unit) in enumerate(fields):
            tk.Label(input_frame, text=label, bg="#2d2d44", fg="white",
                     font=("Segoe UI", 9), anchor="w", width=18).grid(row=i, column=0, pady=4, sticky="w")
            e = tk.Entry(input_frame, font=("Segoe UI", 10), width=10,
                         bg="#3d3d5c", fg="white", insertbackground="white",
                         relief="flat", bd=4)
            e.insert(0, default)
            e.grid(row=i, column=1, pady=4, padx=(5, 2))
            if unit:
                tk.Label(input_frame, text=unit, bg="#2d2d44", fg="#aaaaaa",
                         font=("Segoe UI", 8)).grid(row=i, column=2, padx=3)
            self.entries[label] = e

        # predict button
        tk.Button(
            input_frame,
            text="  Run Prediction  ",
            font=("Segoe UI", 11, "bold"),
            bg="#2a9d8f", fg="white",
            activebackground="#21867a",
            relief="flat", cursor="hand2",
            command=self._run_prediction
        ).grid(row=len(fields), column=0, columnspan=3, pady=(18, 4), sticky="ew")

        # reset button
        tk.Button(
            input_frame,
            text="Reset",
            font=("Segoe UI", 9),
            bg="#444466", fg="white",
            activebackground="#333355",
            relief="flat", cursor="hand2",
            command=self._reset_fields
        ).grid(row=len(fields)+1, column=0, columnspan=3, sticky="ew")

        # ── Output Panel ──
        output_frame = tk.Frame(tab, bg="#1e1e2e")
        output_frame.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=10)

        # Decision Tree result
        self._result_card(output_frame, "🌿  Recommended Crop", "#264653", 0, "dt_label")

        # KMeans result
        self._result_card(output_frame, "🗺  Soil Zone Cluster", "#264653", 1, "km_label")
        self.km_guidance = tk.Label(
            output_frame, text="", bg="#1e1e2e", fg="#aaaaaa",
            font=("Segoe UI", 9, "italic"), wraplength=480, justify="left"
        )
        self.km_guidance.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 8))

        # Linear Regression result
        self._result_card(output_frame, "📈  Predicted Yield", "#264653", 3, "lr_label")

        # confidence note
        self.lr_note = tk.Label(
            output_frame, text="", bg="#1e1e2e", fg="#aaaaaa",
            font=("Segoe UI", 9, "italic")
        )
        self.lr_note.grid(row=4, column=0, sticky="w", padx=20)

    def _result_card(self, parent, title, bg, row, attr_name):
        card = tk.LabelFrame(
            parent, text=f"  {title}  ",
            bg=bg, fg="#2a9d8f",
            font=("Segoe UI", 10, "bold"),
            padx=10, pady=8
        )
        card.grid(row=row, column=0, sticky="ew", padx=10, pady=4)
        parent.columnconfigure(0, weight=1)
        label = tk.Label(card, text="—", bg=bg, fg="white",
                         font=("Segoe UI", 13, "bold"))
        label.pack(anchor="w")
        setattr(self, attr_name, label)

    def _run_prediction(self):
        try:
            keys = list(self.entries.keys())
            values = [float(self.entries[k].get()) for k in keys]
        except ValueError:
            messagebox.showerror("Input Error", "All fields must be numeric.")
            return

        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        # Decision Tree
        crop_encoded = dt_model.predict(arr_scaled)[0]
        crop_name = label_enc.inverse_transform([crop_encoded])[0].capitalize()
        self.dt_label.config(text=crop_name)

        # KMeans
        cluster = km_model.predict(arr_scaled)[0]
        self.km_label.config(text=f"Cluster {cluster}")
        self.km_guidance.config(text=get_cluster_guidance(cluster))

        # Linear Regression
        yield_pred = lr_model.predict(arr_scaled)[0]
        margin = yield_pred * 0.08
        self.lr_label.config(text=f"{yield_pred:,.0f} hg/ha")
        self.lr_note.config(
            text=f"Confidence bound: ± {margin:,.0f} hg/ha  "
                 f"({(yield_pred - margin):,.0f} – {(yield_pred + margin):,.0f})"
        )

    def _reset_fields(self):
        defaults = ["0", "0", "0", "25", "70", "6.5", "200"]
        for key, val in zip(self.entries, defaults):
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, val)
        self.dt_label.config(text="—")
        self.km_label.config(text="—")
        self.km_guidance.config(text="")
        self.lr_label.config(text="—")
        self.lr_note.config(text="")

    # ──────────────────────────────────────────
    # TAB 2 - VISUALIZATIONS
    # ──────────────────────────────────────────

    def _build_plots_tab(self):
        tab = self.tab_plots

        tk.Label(tab, text="Model Evaluation Plots",
                 bg="#1e1e2e", fg="#2a9d8f",
                 font=("Segoe UI", 12, "bold")).pack(pady=(10, 4))

        plot_frame = tk.Frame(tab, bg="#1e1e2e")
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        plot_files = [
            ("results/feature_importance.png", "Feature Importance (Decision Tree)"),
            ("results/cluster_scatter.png",    "Cluster Scatter (KMeans)"),
            ("results/residual_plot.png",       "Residual Plot (Linear Regression)"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor("#1e1e2e")

        for ax, (path, title) in zip(axes, plot_files):
            full_path = os.path.join(BASE, path)
            if os.path.exists(full_path):
                img = plt.imread(full_path)
                ax.imshow(img)
                ax.set_title(title, color="white", fontsize=8, pad=4)
            else:
                ax.text(0.5, 0.5, "Plot not found.\nRun models.py first.",
                        ha="center", va="center", color="gray")
            ax.axis("off")
            ax.set_facecolor("#1e1e2e")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ──────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────

if __name__ == "__main__":
    app = CropIQApp()
    app.mainloop()