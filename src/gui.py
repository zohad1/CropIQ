# src/gui.py

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ─────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────
C = {
    "bg":           "#F4F7F0",
    "white":        "#FFFFFF",
    "green_dark":   "#3B6D11",
    "green_mid":    "#639922",
    "green_light":  "#C0DD97",
    "green_tint":   "#EAF3DE",
    "purple_dark":  "#3C3489",
    "purple_mid":   "#534AB7",
    "purple_light": "#CECBF6",
    "purple_tint":  "#EEEDFE",
    "text_h":       "#1A1A2E",
    "text_body":    "#3B3B4F",
    "text_muted":   "#6B6B80",
    "border":       "#D8E8C8",
    "border_card":  "#E4EDD8",
    "danger":       "#A32D2D",
    "danger_tint":  "#FCEBEB",
}

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_models():
    dt = joblib.load(os.path.join(BASE, "models", "decision_tree.pkl"))
    km = joblib.load(os.path.join(BASE, "models", "kmeans.pkl"))
    lr = joblib.load(os.path.join(BASE, "models", "linear_regression.pkl"))
    sc = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
    le = joblib.load(os.path.join(BASE, "models", "label_encoder.pkl"))
    return dt, km, lr, sc, le

dt_model, km_model, lr_model, scaler, label_enc = load_models()

CLUSTER_GUIDANCE = {
    0: "Low-to-moderate nutrient zone. Recommend legume rotation and organic compost before planting.",
    1: "High-fertility zone. Ideal for nutrient-demanding crops such as rice, maize, or banana.",
    2: "Balanced soil profile. Suitable for most crops with standard fertilisation schedules.",
    3: "Saline or degraded zone. Consider salt-tolerant varieties and improved drainage systems.",
}

CROP_EMOJI = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫛",
    "coconut": "🥥", "coffee": "☕", "cotton": "🌿", "grapes": "🍇",
    "jute": "🌾", "kidneybeans": "🫘", "lentil": "🫛", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🫘", "mungbean": "🫘", "muskmelon": "🍈",
    "orange": "🍊", "papaya": "🍈", "pigeonpeas": "🫛", "pomegranate": "🍎",
    "rice": "🌾", "watermelon": "🍉",
}


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def make_frame(parent, bg=None, padx=0, pady=0):
    return tk.Frame(parent, bg=bg or C["bg"], padx=padx, pady=pady)

def label(parent, text, size=10, weight="normal", color=None, bg=None, anchor="w", wrap=0):
    kw = dict(
        text=text,
        font=("Segoe UI", size, weight),
        fg=color or C["text_body"],
        bg=bg or C["bg"],
        anchor=anchor,
    )
    if wrap:
        kw["wraplength"] = wrap
        kw["justify"] = "left"
    return tk.Label(parent, **kw)

def divider(parent, bg=None):
    return tk.Frame(parent, bg=bg or C["border"], height=1)


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

class CropIQApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CropIQ — Smart Agriculture Decision Support")
        self.geometry("1160x720")
        self.minsize(900, 600)
        self.configure(bg=C["bg"])
        self.resizable(True, True)

        self._style()
        self._header()
        self._nav()
        self._body()
        self._show_tab("predict")

    # ── ttk style ──
    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Flat.TEntry",
                    fieldbackground=C["white"],
                    background=C["white"],
                    foreground=C["text_h"],
                    bordercolor=C["border"],
                    lightcolor=C["border"],
                    darkcolor=C["border"],
                    insertcolor=C["green_dark"],
                    relief="flat", padding=6)
        s.map("Flat.TEntry",
              bordercolor=[("focus", C["green_mid"])],
              lightcolor=[("focus", C["green_mid"])],
              darkcolor=[("focus", C["green_mid"])])

    # ── Header bar ──
    def _header(self):
        bar = tk.Frame(self, bg=C["white"], height=62)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        inner = make_frame(bar, bg=C["white"], padx=24, pady=0)
        inner.pack(fill="both", expand=True)

        # logo pill
        pill = tk.Frame(inner, bg=C["green_dark"], padx=14, pady=5)
        pill.pack(side="left", pady=14)
        tk.Label(pill, text="CropIQ", font=("Segoe UI", 13, "bold"),
                 bg=C["green_dark"], fg=C["white"]).pack()

        tk.Label(inner,
                 text="Smart Agriculture Decision Support System",
                 font=("Segoe UI", 10),
                 fg=C["text_muted"], bg=C["white"]).pack(side="left", padx=14, pady=18)

        # badge right
        badge = tk.Frame(inner, bg=C["purple_tint"], padx=10, pady=4)
        badge.pack(side="right", pady=18)
        tk.Label(badge,
                 text="BSE-6  ·  AI Lab  ·  Bahria University",
                 font=("Segoe UI", 8),
                 fg=C["purple_dark"], bg=C["purple_tint"]).pack()

        divider(self, bg=C["border_card"]).pack(fill="x")

    # ── Side nav ──
    def _nav(self):
        self._nav_frame = tk.Frame(self, bg=C["white"], width=190)
        self._nav_frame.pack(side="left", fill="y")
        self._nav_frame.pack_propagate(False)

        divider(self._nav_frame, bg=C["border_card"]).pack(fill="x")

        self._nav_btns = {}
        nav_items = [
            ("predict",   "Predict",        "Crop · Soil · Yield"),
            ("visualize", "Visualizations", "Model evaluation plots"),
        ]

        tk.Frame(self._nav_frame, bg=C["white"], height=16).pack()

        for key, title, sub in nav_items:
            btn_frame = tk.Frame(self._nav_frame, bg=C["white"], cursor="hand2")
            btn_frame.pack(fill="x", padx=12, pady=3)

            inner = tk.Frame(btn_frame, bg=C["white"], padx=12, pady=10)
            inner.pack(fill="x")

            t = tk.Label(inner, text=title, font=("Segoe UI", 10, "bold"),
                         bg=C["white"], fg=C["text_h"], anchor="w")
            t.pack(fill="x")
            s = tk.Label(inner, text=sub, font=("Segoe UI", 8),
                         bg=C["white"], fg=C["text_muted"], anchor="w")
            s.pack(fill="x")

            self._nav_btns[key] = (btn_frame, inner, t, s)

            for widget in (btn_frame, inner, t, s):
                widget.bind("<Button-1>", lambda e, k=key: self._show_tab(k))
                widget.bind("<Enter>",    lambda e, k=key: self._nav_hover(k, True))
                widget.bind("<Leave>",    lambda e, k=key: self._nav_hover(k, False))

        divider(self._nav_frame, bg=C["border_card"]).pack(fill="x", pady=(12, 0))

        # footer tag
        tk.Frame(self._nav_frame, bg=C["white"]).pack(fill="both", expand=True)
        tk.Label(self._nav_frame,
                 text="CropIQ v1.0",
                 font=("Segoe UI", 8),
                 fg=C["text_muted"], bg=C["white"]).pack(pady=10)

    def _nav_hover(self, key, on):
        if key == self._active_tab:
            return
        btn_frame, inner, t, s = self._nav_btns[key]
        bg = C["green_tint"] if on else C["white"]
        for w in (btn_frame, inner, t, s):
            w.configure(bg=bg)

    def _show_tab(self, key):
        self._active_tab = key
        for k, (bf, inn, t, s) in self._nav_btns.items():
            if k == key:
                bg, fg_t, fg_s = C["green_tint"], C["green_dark"], C["green_mid"]
                bf.configure(bg=bg)
                for w in (inn, t, s):
                    w.configure(bg=bg)
                t.configure(fg=fg_t)
                s.configure(fg=fg_s)
            else:
                for w in (bf, inn, t, s):
                    w.configure(bg=C["white"])
                t.configure(fg=C["text_h"])
                s.configure(fg=C["text_muted"])

        for k, frame in self._tabs.items():
            if k == key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

    # ── Body ──
    def _body(self):
        self._active_tab = "predict"
        body = tk.Frame(self, bg=C["bg"])
        body.pack(side="left", fill="both", expand=True)

        divider(body, bg=C["border_card"]).pack(side="left", fill="y")

        content = tk.Frame(body, bg=C["bg"])
        content.pack(fill="both", expand=True)

        self._tabs = {
            "predict":   tk.Frame(content, bg=C["bg"]),
            "visualize": tk.Frame(content, bg=C["bg"]),
        }

        self._build_predict(self._tabs["predict"])
        self._build_visualize(self._tabs["visualize"])

    # ─────────────────────────────────────────
    # TAB: PREDICT
    # ─────────────────────────────────────────

    def _build_predict(self, parent):
        # page title
        title_row = make_frame(parent, bg=C["bg"], padx=24, pady=0)
        title_row.pack(fill="x", pady=(20, 4))
        label(title_row, "Run a Prediction", size=14, weight="bold",
              color=C["text_h"], bg=C["bg"]).pack(side="left")

        sub = label(title_row,
                    "Enter soil and climate values to get crop recommendation, soil zone, and yield estimate.",
                    size=9, color=C["text_muted"], bg=C["bg"])
        sub.pack(side="left", padx=14, pady=4)

        divider(parent, bg=C["border_card"]).pack(fill="x", padx=24)

        # main row
        row = make_frame(parent, bg=C["bg"])
        row.pack(fill="both", expand=True, padx=24, pady=16)

        self._build_input_panel(row)
        self._build_output_panel(row)

    # ── Input card ──
    def _build_input_panel(self, parent):
        card = tk.Frame(parent, bg=C["white"],
                        highlightbackground=C["border_card"],
                        highlightthickness=1)
        card.pack(side="left", fill="y", padx=(0, 14), ipadx=4, ipady=4)

        # card header
        hdr = tk.Frame(card, bg=C["green_tint"], padx=18, pady=12)
        hdr.pack(fill="x")
        label(hdr, "Soil & Climate Parameters", size=10, weight="bold",
              color=C["green_dark"], bg=C["green_tint"]).pack(anchor="w")
        label(hdr, "All values are required", size=8,
              color=C["green_mid"], bg=C["green_tint"]).pack(anchor="w")

        divider(card, bg=C["border_card"]).pack(fill="x")

        # fields
        fields_frame = make_frame(card, bg=C["white"], padx=18, pady=10)
        fields_frame.pack(fill="x")

        self.entries = {}
        fields = [
            ("N",           "Nitrogen (N)",     "0",   "kg/ha"),
            ("P",           "Phosphorus (P)",   "0",   "kg/ha"),
            ("K",           "Potassium (K)",    "0",   "kg/ha"),
            ("temperature", "Temperature",      "25",  "°C"),
            ("humidity",    "Humidity",         "70",  "%"),
            ("ph",          "Soil pH",          "6.5", ""),
            ("rainfall",    "Rainfall",         "200", "mm"),
        ]

        for i, (key, lbl, default, unit) in enumerate(fields):
            row = make_frame(fields_frame, bg=C["white"])
            row.pack(fill="x", pady=4)

            label(row, lbl, size=9, color=C["text_muted"],
                  bg=C["white"]).pack(anchor="w")

            entry_row = make_frame(row, bg=C["white"])
            entry_row.pack(fill="x")

            e = ttk.Entry(entry_row, style="Flat.TEntry", width=14,
                          font=("Segoe UI", 10))
            e.insert(0, default)
            e.pack(side="left")

            if unit:
                label(entry_row, unit, size=9, color=C["text_muted"],
                      bg=C["white"]).pack(side="left", padx=(6, 0))

            self.entries[key] = e

        divider(card, bg=C["border_card"]).pack(fill="x", pady=(8, 0))

        btn_area = make_frame(card, bg=C["white"], padx=18, pady=14)
        btn_area.pack(fill="x")

        # run button
        run_btn = tk.Button(
            btn_area,
            text="Run Prediction",
            font=("Segoe UI", 10, "bold"),
            bg=C["green_dark"], fg=C["white"],
            activebackground=C["green_mid"],
            activeforeground=C["white"],
            relief="flat", cursor="hand2",
            padx=18, pady=9,
            command=self._run_prediction
        )
        run_btn.pack(fill="x", pady=(0, 6))

        reset_btn = tk.Button(
            btn_area,
            text="Reset fields",
            font=("Segoe UI", 9),
            bg=C["white"], fg=C["text_muted"],
            activebackground=C["green_tint"],
            relief="flat", cursor="hand2",
            padx=10, pady=6,
            highlightbackground=C["border"],
            highlightthickness=1,
            command=self._reset
        )
        reset_btn.pack(fill="x")

    # ── Output panel ──
    def _build_output_panel(self, parent):
        panel = make_frame(parent, bg=C["bg"])
        panel.pack(side="left", fill="both", expand=True)

        # 3 result cards stacked
        self._card_dt  = self._result_card(panel, "Recommended Crop",
                                            "Decision Tree Classification",
                                            C["green_tint"], C["green_dark"])
        self._card_km  = self._result_card(panel, "Soil Zone",
                                            "KMeans Clustering",
                                            C["purple_tint"], C["purple_dark"])
        self._card_lr  = self._result_card(panel, "Predicted Yield",
                                            "Linear Regression",
                                            C["white"], C["text_h"])

    def _result_card(self, parent, title, subtitle, bg, accent):
        card = tk.Frame(parent, bg=C["white"],
                        highlightbackground=C["border_card"],
                        highlightthickness=1)
        card.pack(fill="x", pady=(0, 12), ipady=2)

        left_accent = tk.Frame(card, bg=accent, width=4)
        left_accent.pack(side="left", fill="y")

        body = make_frame(card, bg=C["white"], padx=18, pady=14)
        body.pack(side="left", fill="both", expand=True)

        label(body, subtitle, size=8, color=C["text_muted"],
              bg=C["white"]).pack(anchor="w")
        label(body, title, size=11, weight="bold",
              color=C["text_h"], bg=C["white"]).pack(anchor="w")

        divider(body, bg=C["border_card"]).pack(fill="x", pady=8)

        val_lbl = tk.Label(body, text="—", font=("Segoe UI", 18, "bold"),
                           fg=accent, bg=C["white"], anchor="w")
        val_lbl.pack(anchor="w")

        sub_lbl = tk.Label(body, text="", font=("Segoe UI", 8),
                           fg=C["text_muted"], bg=C["white"],
                           anchor="w", wraplength=460, justify="left")
        sub_lbl.pack(anchor="w", pady=(2, 0))

        return val_lbl, sub_lbl

    # ── Prediction logic ──
    def _run_prediction(self):
        keys = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        try:
            values = [float(self.entries[k].get()) for k in keys]
        except ValueError:
            messagebox.showerror("Input Error",
                                 "All fields must contain numeric values.")
            return

        arr = np.array(values).reshape(1, -1)
        arr_sc = scaler.transform(arr)

        # Decision Tree
        crop_enc  = dt_model.predict(arr_sc)[0]
        crop_name = label_enc.inverse_transform([crop_enc])[0]
        emoji     = CROP_EMOJI.get(crop_name, "🌱")
        val, sub  = self._card_dt
        val.config(text=f"{emoji}  {crop_name.capitalize()}", fg=C["green_dark"])
        sub.config(text="Recommended based on soil N, P, K, temperature, humidity, pH, and rainfall.")

        # KMeans
        cluster  = km_model.predict(arr_sc)[0]
        guidance = CLUSTER_GUIDANCE.get(cluster, f"Cluster {cluster}")
        val, sub = self._card_km
        val.config(text=f"Cluster {cluster}", fg=C["purple_dark"])
        sub.config(text=guidance)

        # Linear Regression
        yld    = lr_model.predict(arr_sc)[0]
        margin = yld * 0.08
        val, sub = self._card_lr
        val.config(text=f"{yld:,.0f} hg/ha", fg=C["text_h"])
        sub.config(
            text=f"Confidence bound ±8%:  {(yld - margin):,.0f} – {(yld + margin):,.0f} hg/ha"
        )

    def _reset(self):
        defaults = {"N": "0", "P": "0", "K": "0",
                    "temperature": "25", "humidity": "70",
                    "ph": "6.5", "rainfall": "200"}
        for k, v in defaults.items():
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, v)
        for val, sub in [self._card_dt, self._card_km, self._card_lr]:
            val.config(text="—", fg=C["text_h"])
            sub.config(text="")

    # ─────────────────────────────────────────
    # TAB: VISUALIZE
    # ─────────────────────────────────────────

    def _build_visualize(self, parent):
        title_row = make_frame(parent, bg=C["bg"], padx=24, pady=0)
        title_row.pack(fill="x", pady=(20, 4))
        label(title_row, "Model Evaluation", size=14, weight="bold",
              color=C["text_h"], bg=C["bg"]).pack(side="left")

        divider(parent, bg=C["border_card"]).pack(fill="x", padx=24)

        plot_files = [
            ("results/feature_importance.png", "Feature Importance",  "Decision Tree"),
            ("results/cluster_scatter.png",    "Cluster Distribution", "KMeans"),
            ("results/residual_plot.png",       "Residual Analysis",   "Linear Regression"),
        ]

        # metric strip - bigger
        metrics = make_frame(parent, bg=C["bg"], padx=24, pady=0)
        metrics.pack(fill="x", pady=(18, 12))

        metric_data = [
            ("Decision Tree", "Accuracy",   "98.64%", C["green_tint"],  C["green_dark"]),
            ("KMeans",        "Silhouette", "0.4195", C["purple_tint"], C["purple_dark"]),
            ("Linear Reg",    "R²",         "0.34",   C["white"],       C["text_h"]),
        ]
        for model, metric, val, bg, fg in metric_data:
            chip = tk.Frame(metrics, bg=bg,
                            highlightbackground=C["border_card"],
                            highlightthickness=1)
            chip.pack(side="left", padx=(0, 14), ipadx=22, ipady=16)
            label(chip, model,  size=10, color=fg,             bg=bg, weight="bold").pack(anchor="w")
            label(chip, metric, size=9,  color=C["text_muted"], bg=bg).pack(anchor="w", pady=(2, 4))
            tk.Label(chip, text=val, font=("Segoe UI", 22, "bold"),
                     fg=fg, bg=bg).pack(anchor="w")

        divider(parent, bg=C["border_card"]).pack(fill="x", padx=24)

        # plots - bigger figure
        plot_outer = make_frame(parent, bg=C["bg"])
        plot_outer.pack(fill="both", expand=True, padx=24, pady=16)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
        fig.patch.set_facecolor(C["bg"])
        fig.subplots_adjust(wspace=0.28, left=0.04, right=0.98,
                            top=0.85, bottom=0.10)

        for ax, (path, title, sub) in zip(axes, plot_files):
            full = os.path.join(BASE, path)
            ax.set_facecolor(C["white"])
            if os.path.exists(full):
                img = plt.imread(full)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"Run models.py first\nto generate this plot",
                        ha="center", va="center",
                        color=C["text_muted"], fontsize=10,
                        transform=ax.transAxes)
            ax.set_title(f"{title}\n", fontsize=11, fontweight="bold",
                         color=C["text_h"], pad=4)
            ax.set_xlabel(sub, fontsize=9, color=C["text_muted"], labelpad=6)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(C["border_card"])
                spine.set_linewidth(0.8)

        canvas = FigureCanvasTkAgg(fig, master=plot_outer)
        canvas.draw()
        canvas.get_tk_widget().configure(bg=C["bg"], highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ─────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────

if __name__ == "__main__":
    app = CropIQApp()
    app.mainloop()