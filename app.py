import collections
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.compose._column_transformer as ct
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

# Model was pickled with sklearn 1.6.1; newer versions removed this private class
if not hasattr(ct, "_RemainderColsList"):
    ct._RemainderColsList = type("_RemainderColsList", (collections.UserList,), {})
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

MODEL_PATH = Path(__file__).parent / "diabetes_rf_model.joblib"

st.set_page_config(
    page_title="GlucoSense · Diabetes Risk AI",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# (label, unit, min, max, default, step, normal_low, normal_high, group)
NUMERIC_SPECS = {
    "HbA1c": ("HbA1c", "%", 3.0, 15.0, 5.4, 0.1, 4.0, 5.7, "Glycemic"),
    "Fasting_Blood_Glucose": ("Fasting Glucose", "mg/dL", 50.0, 300.0, 92.0, 1.0, 70.0, 99.0, "Glycemic"),
    "Postprandial_Blood_Glucose": ("Postprandial Glucose", "mg/dL", 70.0, 400.0, 125.0, 1.0, 70.0, 140.0, "Glycemic"),
    "Random_Blood_Glucose": ("Random Glucose", "mg/dL", 70.0, 400.0, 120.0, 1.0, 70.0, 140.0, "Glycemic"),
    "OGTT": ("OGTT (2-hr)", "mg/dL", 70.0, 400.0, 120.0, 1.0, 70.0, 140.0, "Glycemic"),
    "Fructosamine_Levels": ("Fructosamine", "µmol/L", 100.0, 600.0, 240.0, 1.0, 200.0, 285.0, "Glycemic"),
    "Insulin_Levels": ("Insulin", "µU/mL", 1.0, 100.0, 10.0, 0.5, 2.0, 25.0, "Metabolic"),
    "HOMA_IR": ("HOMA-IR", "", 0.1, 20.0, 1.8, 0.1, 0.5, 2.5, "Metabolic"),
    "C_Peptide": ("C-Peptide", "ng/mL", 0.1, 10.0, 1.5, 0.1, 0.5, 2.0, "Metabolic"),
    "Triglyceride_Levels": ("Triglycerides", "mg/dL", 30.0, 600.0, 130.0, 1.0, 30.0, 150.0, "Metabolic"),
    "HDL_Cholesterol": ("HDL Cholesterol", "mg/dL", 10.0, 120.0, 55.0, 1.0, 40.0, 120.0, "Metabolic"),
    "BMI": ("BMI", "kg/m²", 10.0, 60.0, 23.5, 0.1, 18.5, 24.9, "Body & Vitals"),
    "Waist_Circumference": ("Waist Circumference", "in", 20.0, 70.0, 34.0, 0.5, 20.0, 40.0, "Body & Vitals"),
    "Blood_Pressure_Systolic": ("Systolic BP", "mmHg", 70.0, 220.0, 118.0, 1.0, 90.0, 120.0, "Body & Vitals"),
    "Blood_Pressure_Diastolic": ("Diastolic BP", "mmHg", 40.0, 140.0, 76.0, 1.0, 60.0, 80.0, "Body & Vitals"),
}

CATEGORICAL_LABELS = {
    "Family_History_of_Diabetes": ("Family history of diabetes", "🧬"),
    "Hypertension": ("Hypertension", "💓"),
    "Physical_Activity": ("Physically active", "🏃"),
    "Obesity": ("Obesity", "⚖️"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(circle at 10% 0%, #1e1b4b 0%, #0f172a 45%, #020617 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85);
    border-right: 1px solid rgba(148, 163, 184, 0.15);
}
[data-testid="stHeader"] { background: transparent; }
.card {
    background: rgba(30, 41, 59, 0.55); backdrop-filter: blur(12px);
    border: 1px solid rgba(148,163,184,.18); border-radius: 18px;
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}
.result {
    border-radius: 20px; padding: 1.6rem; text-align: center;
    animation: pop .5s ease-out;
}
.result.pos { background: linear-gradient(135deg, rgba(239,68,68,.35), rgba(244,114,182,.2)); border: 1px solid rgba(248,113,113,.6); box-shadow: 0 0 40px -10px #ef4444; }
.result.neg { background: linear-gradient(135deg, rgba(16,185,129,.35), rgba(56,189,248,.2)); border: 1px solid rgba(52,211,153,.6); box-shadow: 0 0 40px -10px #10b981; }
.result .label { font-size: .85rem; letter-spacing: .2em; text-transform: uppercase; color: #cbd5e1; }
.result .value { font-size: 2.4rem; font-weight: 800; margin: .3rem 0; }
.result .sub { color: #e2e8f0; opacity: .85; }
@keyframes pop { 0% { transform: scale(.95); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
.metric-pill {
    display: inline-block; padding: .35rem .8rem; margin: .2rem; border-radius: 999px;
    font-size: .82rem; border: 1px solid rgba(148,163,184,.25); background: rgba(15,23,42,.6);
}
.metric-pill.bad { border-color: rgba(248,113,113,.6); color: #fca5a5; }
.metric-pill.ok { border-color: rgba(52,211,153,.5); color: #6ee7b7; }
.group-title { font-weight: 600; color: #a5b4fc; margin: 1rem 0 .2rem 0; letter-spacing: .03em; }
.footer { text-align: center; color: #64748b; font-size: .8rem; margin-top: 2rem; }
</style>
"""


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def init_state(artifact):
    for key, spec in NUMERIC_SPECS.items():
        st.session_state.setdefault(key, spec[4])
    for key in artifact["categorical_features"]:
        st.session_state.setdefault(key, "No")


def predict(artifact, record: dict) -> dict:
    input_df = pd.DataFrame([record])[artifact["features"]]
    proba = artifact["model"].predict_proba(input_df)[0]
    idx = int(np.argmax(proba))
    classes = artifact["label_encoder"].classes_
    return {
        "prediction": classes[idx],
        "confidence": float(proba[idx]),
        "probabilities": dict(zip(classes, map(float, proba))),
    }


def gauge(prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44, "color": "#f8fafc"}},
        title={"text": "Diabetes probability", "font": {"size": 16, "color": "#cbd5e1"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar": {"color": "#f472b6" if prob >= 0.5 else "#34d399", "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(16,185,129,.25)"},
                {"range": [30, 60], "color": "rgba(234,179,8,.25)"},
                {"range": [60, 100], "color": "rgba(239,68,68,.25)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": 50},
        },
    ))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})
    return fig


def importance_chart(artifact) -> go.Figure:
    names = artifact["numeric_features"] + artifact["categorical_features"]
    imp = pd.Series(artifact["model"].named_steps["rf"].feature_importances_, index=names).sort_values()
    labels = [NUMERIC_SPECS[n][0] if n in NUMERIC_SPECS else CATEGORICAL_LABELS[n][0] for n in imp.index]
    fig = go.Figure(go.Bar(
        x=imp.values, y=labels, orientation="h",
        marker=dict(color=imp.values, colorscale=[[0, "#38bdf8"], [0.5, "#a855f7"], [1, "#f472b6"]]),
    ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#cbd5e1"}, xaxis=dict(gridcolor="rgba(148,163,184,.15)", title="Importance"))
    return fig


def radar_chart(record: dict) -> go.Figure:
    keys = ["HbA1c", "Fasting_Blood_Glucose", "Postprandial_Blood_Glucose", "OGTT",
            "HOMA_IR", "Triglyceride_Levels", "BMI", "Blood_Pressure_Systolic"]
    # Ratio to upper normal bound: 1.0 marks the edge of the reference range
    ratios = [record[k] / NUMERIC_SPECS[k][7] for k in keys]
    labels = [NUMERIC_SPECS[k][0] for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[1] * len(keys) + [1], theta=labels + labels[:1], name="Normal limit",
                                  line=dict(color="#34d399", dash="dash"), fill="toself",
                                  fillcolor="rgba(52,211,153,.08)"))
    fig.add_trace(go.Scatterpolar(r=ratios + ratios[:1], theta=labels + labels[:1], name="Patient",
                                  line=dict(color="#f472b6", width=3), fill="toself",
                                  fillcolor="rgba(244,114,182,.25)"))
    fig.update_layout(
        height=420, margin=dict(l=40, r=40, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"}, legend=dict(orientation="h", y=-0.1),
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(gridcolor="rgba(148,163,184,.2)", showticklabels=False),
                   angularaxis=dict(gridcolor="rgba(148,163,184,.2)")),
    )
    return fig


def render_inputs(group: str):
    keys = [k for k, s in NUMERIC_SPECS.items() if s[8] == group]
    cols = st.columns(2)
    for i, key in enumerate(keys):
        label, unit, lo, hi, _, step, nlo, nhi, _ = NUMERIC_SPECS[key]
        with cols[i % 2]:
            st.slider(f"{label} ({unit})" if unit else label, lo, hi, step=step, key=key,
                      help=f"Reference range: {nlo:g} – {nhi:g} {unit}")


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        st.stop()
    artifact = load_artifact()
    init_state(artifact)

    with st.sidebar:
        st.markdown("## 🩸 GlucoSense")
        st.caption("Random Forest diabetes risk estimator")
        st.markdown("### 📊 Model card")
        m1, m2 = st.columns(2)
        m1.metric("CV accuracy", f"{artifact['cv_accuracy'] * 100:.1f}%")
        m2.metric("Test accuracy", f"{artifact['test_accuracy'] * 100:.1f}%")
        st.metric("Features used", len(artifact["features"]))

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown("### 🧪 Patient profile")
        for title, group in [("🍬 Glycemic", "Glycemic"), ("🔬 Metabolic", "Metabolic"),
                             ("🫀 Body & Vitals", "Body & Vitals")]:
            st.markdown(f'<div class="group-title">{title}</div>', unsafe_allow_html=True)
            render_inputs(group)
        st.markdown('<div class="group-title">🧬 History & Lifestyle</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, key in enumerate(artifact["categorical_features"]):
            label, icon = CATEGORICAL_LABELS.get(key, (key.replace("_", " "), "•"))
            options = artifact["categorical_values"].get(key, ["No", "Yes"])
            with cols[i % 2]:
                st.radio(f"{icon} {label}", options, key=key, horizontal=True)

    record = {f: st.session_state[f] for f in artifact["features"]}
    result = predict(artifact, record)
    positive_label = next((c for c in result["probabilities"] if str(c).lower().startswith("pos")),
                          list(result["probabilities"])[-1])
    p_pos = result["probabilities"][positive_label]
    is_pos = result["prediction"] == positive_label

    with right:
        st.markdown("### 🎯 Prediction")
        css_class = "pos" if is_pos else "neg"
        headline = "Diabetes likely" if is_pos else "Diabetes unlikely"
        st.markdown(
            f'<div class="result {css_class}"><div class="label">Model verdict</div>'
            f'<div class="value">{"⚠️" if is_pos else "✅"} {headline}</div>'
            f'<div class="sub">Predicted <b>{result["prediction"]}</b> with '
            f'<b>{result["confidence"] * 100:.1f}%</b> confidence</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge(p_pos), width="stretch", config={"displayModeBar": False})

        flags = []
        for key, (label, unit, *_r, nlo, nhi, _g) in NUMERIC_SPECS.items():
            val = record[key]
            ok = nlo <= val <= nhi
            flags.append(f'<span class="metric-pill {"ok" if ok else "bad"}">'
                         f'{"●" if ok else "▲"} {label}: {val:g}</span>')
        st.markdown('<div class="card"><b>Biomarker check</b><br>' + "".join(flags) + "</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### 🕸️ Biomarker radar")
        st.caption("Values scaled to the upper limit of the reference range (dashed = limit).")
        st.plotly_chart(radar_chart(record), width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown("### 🌲 What drives the model")
        st.caption("Random Forest feature importances (mean decrease in impurity).")
        st.plotly_chart(importance_chart(artifact), width="stretch", config={"displayModeBar": False})

    with st.expander("🔎 Raw model input"):
        st.dataframe(pd.DataFrame([record]).T.rename(columns={0: "value"}).astype(str), width="stretch")

    st.markdown('<div class="footer">Built with Streamlit · scikit-learn · Plotly</div>', unsafe_allow_html=True)


main()
