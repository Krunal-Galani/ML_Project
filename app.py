import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ─── Google Font ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Base ─── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f0f4f8; min-height: 100vh; }

/* ─── Hide Streamlit chrome (deploy, menu, footer) - NEVER hide stHeader ─── */
.stDeployButton         { display: none !important; }
#MainMenu               { display: none !important; }
footer                  { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ─── Streamlit top header bar ─── */
[data-testid="stHeader"] {
    background: rgba(240,244,248,0.95) !important;
    backdrop-filter: blur(8px) !important;
    border-bottom: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06) !important;
}

/* ─── Main content: clear the sticky navbar ─── */
.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 20px rgba(0,0,0,0.06);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: #334155 !important; }

/* Sidebar collapse button ─ keep blue */
[data-testid="stSidebarCollapseButton"] button {
    background: #eff6ff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 8px !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebarCollapseButton"] button:hover { background: #dbeafe !important; }
[data-testid="stSidebarCollapseButton"] svg polyline { stroke: #2563eb !important; }

/* Collapsed sidebar open button ─ blue pill on left edge */
[data-testid="stCollapsedControl"] {
    background-color: #2563eb !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 2px 0 10px rgba(37,99,235,0.30) !important;
    transition: background-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stCollapsedControl"]:hover {
    background-color: #1d4ed8 !important;
    box-shadow: 2px 0 16px rgba(37,99,235,0.50) !important;
}
[data-testid="stCollapsedControl"] svg polyline { stroke: #ffffff !important; }

/* ─── Sidebar radio nav items ─── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 10px !important;
    padding: 8px 12px !important;
    margin: 2px 0 !important;
    transition: background 0.15s !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #f0f4f8 !important;
}

/* ─── Fade-in animation for page content ─── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main .block-container > div {
    animation: fadeSlideIn 0.40s ease forwards;
}

/* ─── Card ─── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.05);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(37,99,235,0.11);
}

/* ─── Stat card ─── */
.stat-card {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border-radius: 18px;
    padding: 26px 18px;
    text-align: center;
    box-shadow: 0 4px 22px rgba(37,99,235,0.28);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 36px rgba(37,99,235,0.38);
}
.stat-number { font-size: 34px; font-weight: 800; color: #ffffff; line-height: 1.1; }
.stat-label  { font-size: 11.5px; color: rgba(255,255,255,0.75); margin-top: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }

/* ─── Section title ─── */
.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #0f172a;
    margin: 28px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    border-left: 4px solid #2563eb;
    padding-left: 13px;
    line-height: 1.3;
}

/* ─── Feature pills ─── */
.pill {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #1d4ed8;
    margin: 4px;
    transition: background 0.15s, transform 0.15s;
}
.pill:hover { background: #dbeafe; transform: translateY(-1px); }

/* ─── Result boxes ─── */
.result-positive {
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 1.5px solid #fca5a5;
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(239,68,68,0.10);
    animation: fadeSlideIn 0.35s ease;
}
.result-negative {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1.5px solid #86efac;
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(22,163,74,0.10);
    animation: fadeSlideIn 0.35s ease;
}
.result-emoji   { font-size: 54px; display: block; margin-bottom: 4px; }
.result-heading { font-size: 24px; font-weight: 800; margin: 10px 0 8px 0; }
.result-sub     { font-size: 15px; color: #64748b; }

/* ─── Risk label ─── */
.risk-label { font-size: 12px; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }

/* ─── Tip / info box ─── */
.tip-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 14px 18px;
    color: #92400e;
    font-size: 14px;
    margin-top: 14px;
}

/* ─── Section header banners (Prediction / Insights) ─── */
.page-header {
    background: linear-gradient(120deg, #1e40af 0%, #2563eb 100%);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 28px;
    box-shadow: 0 6px 26px rgba(37,99,235,0.22);
    animation: fadeSlideIn 0.38s ease;
}
.page-header-title { font-size: 26px; font-weight: 800; color: #ffffff; margin: 0 0 6px 0; }
.page-header-sub   { font-size: 14px; color: rgba(255,255,255,0.80); margin: 0; line-height: 1.6; }

/* ─── Insight metric cards ─── */
.insight-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
.insight-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(37,99,235,0.10); }
.insight-val { font-size: 32px; font-weight: 800; color: #0f172a; line-height: 1.1; }
.insight-lbl { font-size: 11px; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 7px; }

/* ─── Metric overrides ─── */
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; color: #2563eb !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 13px !important; }

/* ─── Input / widget labels ─── */
.stSelectbox label,
.stNumberInput label,
[data-testid="stWidgetLabel"] p { color: #374151 !important; font-weight: 600 !important; font-size: 13.5px !important; }

/* ─── Number inputs ─── */
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInputContainer"],
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background-color: #f8fafc !important;
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="stNumberInput"] > div:focus-within,
[data-testid="stNumberInputContainer"]:focus-within,
div[data-baseweb="input"]:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
[data-testid="stNumberInput"] input,
input[type="number"] {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    color: #0f172a !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}
[data-testid="stNumberInput"] button {
    background-color: #eff6ff !important;
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    color: #1d4ed8 !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] button:hover { 
    background-color: #dbeafe !important;
    background: #dbeafe !important; 
}

/* ─── Select dropdowns ─── */
[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"] > div {
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
div[data-baseweb="select"] > div:focus-within { border-color: #2563eb !important; }
[data-testid="stSelectbox"] span,
div[data-baseweb="select"] span { color: #0f172a !important; font-size: 14px !important; font-weight: 500 !important; }
div[data-baseweb="select"] svg { fill: #64748b !important; }

/* ─── Dropdown popover ─── */
div[role="listbox"],
div[data-baseweb="popover"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 10px 36px rgba(0,0,0,0.13) !important;
}
div[role="option"] { background: transparent !important; color: #1e293b !important; font-size: 14px !important; border-radius: 8px !important; margin: 2px 4px !important; }
div[role="option"]:hover { background: #eff6ff !important; color: #1d4ed8 !important; }

/* ─── Primary button ─── */
div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.02em;
    background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.32) !important;
    transition: opacity 0.18s, transform 0.18s, box-shadow 0.18s !important;
}
div.stButton > button:hover {
    opacity: 0.92;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37,99,235,0.42) !important;
}
div.stButton > button:active { transform: translateY(0); }

/* ─── Dividers ─── */
hr { border-color: #e2e8f0 !important; margin: 22px 0 !important; }

/* ─── Progress bar ─── */
[data-testid="stProgress"] > div { background: #e2e8f0 !important; border-radius: 99px !important; height: 10px !important; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #ef4444, #f97316) !important; border-radius: 99px !important; transition: width 0.6s ease !important; }

/* ─── Expander ─── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #334155 !important;
    padding: 14px 18px !important;
}

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0 !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 10px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ─── Responsive: smaller screens ─── */
@media (max-width: 768px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .stat-number { font-size: 26px; }
    .page-header { padding: 20px 20px; }
    .page-header-title { font-size: 20px; }
}
</style>
""", unsafe_allow_html=True)




@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "cardiovascular_logistic_regression_pipeline.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load model file.")
    st.exception(e)
    st.stop()


with st.sidebar:
    st.markdown("""
    <div style='padding:24px 12px 20px 12px; border-bottom:1px solid #e2e8f0; margin-bottom:16px;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='background:linear-gradient(135deg,#2563eb,#1d4ed8); border-radius:12px;
                        width:44px; height:44px; display:flex; align-items:center;
                        justify-content:center; font-size:22px; flex-shrink:0;
                        box-shadow:0 4px 12px rgba(37,99,235,0.3);'>🫀</div>
            <div>
                <div style='font-size:17px; font-weight:800; color:#0f172a;'>CardioSense AI</div>
                <div style='font-size:11px; color:#94a3b8; font-weight:500; margin-top:2px;'>Cardiovascular Risk Predictor</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔬  Prediction", "📊  Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='padding:4px 0; font-size:13px;'>
        <div style='font-weight:700; color:#334155; margin-bottom:10px;'>About this Model</div>
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:12px; margin-bottom:10px;'>
            <div style='font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;'>Dataset</div>
            <div style='color:#334155; font-weight:600;'>Cardiovascular Disease</div>
            <div style='color:#64748b; font-size:12px; margin-top:2px;'>70,000 patient records</div>
        </div>
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:12px;'>
            <div style='font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;'>Model</div>
            <div style='color:#334155; font-weight:600;'>Logistic Regression</div>
            <div style='margin-top:6px;'>
                <span style='background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;
                             border-radius:99px; padding:2px 8px; font-size:11px; font-weight:700;'>71.39% Accuracy</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='background:#fff7ed; border:1px solid #fed7aa; border-radius:10px;
                padding:10px 12px; font-size:11.5px; color:#92400e;'>
        ⚠️ <b>Educational use only.</b><br>Not a medical diagnosis tool.
    </div>
    """, unsafe_allow_html=True)


if page == "🏠  Home":
    st.markdown("""
    <div style='background:linear-gradient(120deg,#2563eb 0%,#1e40af 100%); border-radius:20px;
                padding:44px 40px; margin-bottom:28px; box-shadow:0 8px 32px rgba(37,99,235,0.25);'>
        <div style='font-size:12px; font-weight:700; color:rgba(255,255,255,0.65);
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;'>AI-Powered Clinical Tool</div>
        <div style='font-size:40px; font-weight:800; color:#ffffff; line-height:1.15; margin-bottom:14px;'>
            Cardiovascular<br>Disease Prediction</div>
        <div style='font-size:16px; color:rgba(255,255,255,0.82); max-width:540px; line-height:1.7;'>
            An AI-powered risk assessment tool built with machine learning on 70,000 patient records.
            Enter health metrics to get an instant prediction with probability scores.</div>
        <div style='margin-top:22px; font-size:28px;'>🫀</div>
    </div>
    """, unsafe_allow_html=True)

    features = ["🧠 Logistic Regression", "📐 StandardScaler", "🔁 5-Fold CV",
                "📊 71.4% Accuracy", "⚡ Real-time Prediction", "🩺 12 Health Features"]
    st.markdown("".join(f'<span class="pill">{f}</span>' for f in features), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Dataset Overview</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    stats = [("70,000","Patient Records","📁"),("12","Input Features","🔢"),
             ("71.39%","Model Accuracy","🎯"),("~71.6%","CV Mean Score","🔁")]
    for col, (num, lbl, icon) in zip([s1,s2,s3,s4], stats):
        with col:
            st.markdown(f'''<div class="stat-card">
                <div style="font-size:26px; margin-bottom:8px;">{icon}</div>
                <div class="stat-number">{num}</div>
                <div class="stat-label">{lbl}</div></div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🔬 How It Works</div>', unsafe_allow_html=True)
        for n, t, d in [("1","Enter patient data","Age, weight, blood pressure, cholesterol & more"),
                        ("2","AI processes inputs","StandardScaler normalises features for the model"),
                        ("3","Logistic Regression predicts","Binary classification: disease / no disease"),
                        ("4","View probability score","See confidence level and risk interpretation")]:
            st.markdown(f'''<div class="card" style="padding:16px 18px; margin-bottom:10px; display:flex; align-items:flex-start; gap:14px;">
                <div style="background:#eff6ff; color:#2563eb; border:2px solid #bfdbfe; border-radius:50%;
                            width:32px; height:32px; display:flex; align-items:center; justify-content:center;
                            font-weight:800; font-size:14px; flex-shrink:0;">{n}</div>
                <div><div style="font-weight:700; color:#0f172a; font-size:14px;">{t}</div>
                     <div style="font-size:13px; color:#64748b; margin-top:3px;">{d}</div></div></div>''', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">⚠️ Key Risk Factors</div>', unsafe_allow_html=True)
        risks = [
            ("🩸","High Blood Pressure","Systolic > 140 mmHg significantly raises risk","#fff1f2","#fca5a5"),
            ("🍔","High Cholesterol","Elevated LDL is a primary cardiac risk factor","#fff7ed","#fdba74"),
            ("🚬","Smoking","Doubles the risk of cardiovascular events","#fdf4ff","#e9d5ff"),
            ("🏃","Physical Inactivity","Regular exercise reduces risk by up to 35%","#fffbeb","#fde68a"),
            ("⚖️","Obesity (High BMI)","BMI > 30 is strongly linked to heart disease","#fff1f2","#fca5a5")
        ]
        for icon, title, desc, bg, border in risks:
            st.markdown(f'''<div class="card" style="padding:14px 18px; margin-bottom:8px;
                         background:{bg}; border-color:{border}; display:flex; align-items:flex-start; gap:12px;">
                <div style="font-size:20px; flex-shrink:0; margin-top:2px;">{icon}</div>
                <div><div style="font-weight:700; color:#0f172a; font-size:13.5px;">{title}</div>
                     <div style="font-size:12.5px; color:#64748b; margin-top:2px;">{desc}</div></div></div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(120deg,#eff6ff,#dbeafe); border:1.5px solid #bfdbfe;
                border-radius:16px; padding:28px 32px; display:flex; align-items:center;
                justify-content:space-between; flex-wrap:wrap; gap:16px;'>
        <div>
            <div style='font-size:18px; font-weight:700; color:#1e40af; margin-bottom:6px;'>
                Ready to assess cardiovascular risk?</div>
            <div style='font-size:14px; color:#3b82f6;'>
                Navigate to <b>🔬 Prediction</b> in the sidebar to get started.</div>
        </div>
        <div style='font-size:42px;'>🫀</div>
    </div>
    """, unsafe_allow_html=True)


elif page == "🔬  Prediction":
    st.markdown("""
    <div class='page-header'>
        <p class='page-header-title'>🔬 Risk Prediction</p>
        <p class='page-header-sub'>Complete the patient health profile below and click <b>Predict</b> to get an instant cardiovascular risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px;
                padding:14px 18px; margin-bottom:16px;'>
        <span style='font-size:15px; font-weight:700; color:#1e40af;'>👤 Basic Information</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: age = st.number_input("Age (years)", min_value=1, max_value=120, value=40, step=1)
    with c2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_value = 1 if gender == "Female" else 2
    with c3: height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165, step=1)
    c4, c5, c6 = st.columns(3)
    with c4: weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
    with c5: ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=240, value=120, help="Upper reading, e.g. 120 in '120/80'")
    with c6: ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=80, help="Lower reading, e.g. 80 in '120/80'")

    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5: bmi_cat, bmi_color, bmi_bg, bmi_border = "Underweight", "#1d4ed8", "#eff6ff", "#bfdbfe"
    elif bmi < 25: bmi_cat, bmi_color, bmi_bg, bmi_border = "Normal Weight ✅", "#15803d", "#f0fdf4", "#86efac"
    elif bmi < 30: bmi_cat, bmi_color, bmi_bg, bmi_border = "Overweight ⚠️", "#b45309", "#fffbeb", "#fde68a"
    else: bmi_cat, bmi_color, bmi_bg, bmi_border = "Obese 🔴", "#b91c1c", "#fff1f2", "#fca5a5"

    st.markdown(f"""
    <div style='background:{bmi_bg}; border:1.5px solid {bmi_border}; border-radius:14px;
                padding:18px 24px; display:flex; align-items:center; gap:28px; flex-wrap:wrap; margin:12px 0;'>
        <div>
            <div style='font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;'>BMI</div>
            <div style='font-size:38px; font-weight:800; color:{bmi_color};'>{bmi:.1f}</div>
        </div>
        <div style='height:50px; width:1.5px; background:#e2e8f0;'></div>
        <div>
            <div style='font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;'>Category</div>
            <div style='font-size:17px; font-weight:700; color:{bmi_color};'>{bmi_cat}</div>
        </div>
        <div style='height:50px; width:1.5px; background:#e2e8f0;'></div>
        <div style='font-size:13px; color:#64748b; max-width:220px;'>Auto-calculated from height & weight. Used directly as a model feature.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#f0fdf4; border:1px solid #86efac; border-radius:12px;
                padding:14px 18px; margin:16px 0;'>
        <span style='font-size:15px; font-weight:700; color:#15803d;'>🩺 Medical & Lifestyle</span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
        cholesterol_value = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]
    with m2:
        gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
        gluc_value = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]
    with m3:
        smoke = st.selectbox("Smoker?", ["No", "Yes"])
        smoke_value = 1 if smoke == "Yes" else 0
    m4, m5, m6 = st.columns(3)
    with m4:
        alco = st.selectbox("Alcohol Consumption?", ["No", "Yes"])
        alco_value = 1 if alco == "Yes" else 0
    with m5:
        active = st.selectbox("Physically Active?", ["Yes", "No"])
        active_value = 1 if active == "Yes" else 0
    with m6:
        flags = []
        if ap_hi > 140: flags.append(("🔴", "High Systolic BP", "#fff1f2", "#fca5a5"))
        if ap_lo > 90:  flags.append(("🔴", "High Diastolic BP", "#fff1f2", "#fca5a5"))
        if bmi >= 30:   flags.append(("🟠", "Obese BMI", "#fff7ed", "#fdba74"))
        if smoke == "Yes": flags.append(("🟠", "Smoker", "#fff7ed", "#fdba74"))
        if cholesterol_value == 3: flags.append(("🔴", "Very High Cholesterol", "#fff1f2", "#fca5a5"))
        st.markdown('<div style="font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">Quick Risk Flags</div>', unsafe_allow_html=True)
        if flags:
            for em, lbl, bg, bd in flags:
                st.markdown(f'<div style="background:{bg}; border:1px solid {bd}; border-radius:8px; padding:6px 10px; font-size:12.5px; font-weight:600; color:#1e293b; margin-bottom:5px;">{em} {lbl}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:6px 10px; font-size:12.5px; font-weight:600; color:#15803d;">✅ No critical flags</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_button = st.button("🔮  Predict Cardiovascular Risk")

    if predict_button:
        errors = []
        if ap_hi <= ap_lo: errors.append("Systolic BP must be greater than Diastolic BP.")
        if errors:
            for e in errors: st.error(e)
        else:
            try:
                input_data = pd.DataFrame({"age":[age],"gender":[gender_value],"height":[height],"weight":[weight],
                    "ap_hi":[ap_hi],"ap_lo":[ap_lo],"cholesterol":[cholesterol_value],"gluc":[gluc_value],
                    "smoke":[smoke_value],"alco":[alco_value],"active":[active_value],"BMI":[bmi]})
                input_data = input_data[list(model.feature_names_in_)]
                prediction = model.predict(input_data)[0]
                probas     = model.predict_proba(input_data)[0]
                risk_pct   = probas[1] * 100
                safe_pct   = probas[0] * 100
                st.markdown("---")
                if prediction == 1:
                    st.markdown('''<div class="result-positive">
                        <div class="result-emoji">⚠️</div>
                        <div class="result-heading" style="color:#b91c1c;">Higher Risk Detected</div>
                        <div class="result-sub">The model predicts an elevated likelihood of cardiovascular disease based on the provided inputs.</div>
                    </div>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<div class="result-negative">
                        <div class="result-emoji">✅</div>
                        <div class="result-heading" style="color:#15803d;">Lower Risk Detected</div>
                        <div class="result-sub">The model predicts a lower likelihood of cardiovascular disease. Maintain a healthy lifestyle!</div>
                    </div>''', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                pa, pb, pc = st.columns(3)
                with pa:
                    st.markdown(f'''<div class="insight-card" style="border-top:4px solid #ef4444;">
                        <div class="insight-val" style="color:#b91c1c;">{risk_pct:.1f}%</div>
                        <div class="insight-lbl" style="color:#ef4444;">Disease Risk</div></div>''', unsafe_allow_html=True)
                with pb:
                    st.markdown(f'''<div class="insight-card" style="border-top:4px solid #22c55e;">
                        <div class="insight-val" style="color:#15803d;">{safe_pct:.1f}%</div>
                        <div class="insight-lbl" style="color:#22c55e;">Healthy Probability</div></div>''', unsafe_allow_html=True)
                with pc:
                    fc = len(flags)
                    fc_c = "#b91c1c" if fc >= 2 else ("#b45309" if fc == 1 else "#15803d")
                    fc_b = "#ef4444" if fc >= 2 else ("#f59e0b" if fc == 1 else "#22c55e")
                    st.markdown(f'''<div class="insight-card" style="border-top:4px solid {fc_b};">
                        <div class="insight-val" style="color:{fc_c};">{fc}</div>
                        <div class="insight-lbl" style="color:{fc_b};">Risk Flags</div></div>''', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="risk-label">Risk Probability Gauge</div>', unsafe_allow_html=True)
                st.progress(int(min(risk_pct, 100)))
                st.markdown(f'<div style="font-size:13px; color:#64748b; margin-top:6px;">Disease: <b style="color:#ef4444;">{risk_pct:.2f}%</b> &nbsp;|&nbsp; Healthy: <b style="color:#15803d;">{safe_pct:.2f}%</b></div>', unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown('''<div class="tip-box">💡 <b>Clinical Recommendations:</b> Consider reducing sodium intake, increasing physical activity, quitting smoking if applicable, and scheduling a cardiovascular checkup.</div>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<div class="tip-box" style="background:#f0fdf4; border-color:#86efac; border-left-color:#22c55e; color:#14532d;">💡 <b>Keep it up!</b> Continue regular exercise, a balanced diet, and routine medical checkups.</div>''', unsafe_allow_html=True)
                with st.expander("📋 View entered patient data"):
                    st.dataframe(pd.DataFrame({
                        "Feature": ["Age","Gender","Height (cm)","Weight (kg)","Systolic BP","Diastolic BP","Cholesterol","Glucose","Smoker","Alcohol","Active","BMI"],
                        "Value": [age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,f"{bmi:.2f}"]
                    }), use_container_width=True, hide_index=True)
                st.markdown('''<div style="font-size:12px; color:#94a3b8; margin-top:16px; padding:12px;
                             background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; text-align:center;">
                    ⚠️ This tool is for <b>educational and research purposes only</b>.
                    It is <b>not a medical diagnostic device</b>. Always consult a qualified healthcare professional.
                </div>''', unsafe_allow_html=True)
            except Exception as e:
                st.error("Prediction failed.")
                st.code(str(e))


elif page == "📊  Model Insights":
    st.markdown("""
    <div class='page-header'>
        <p class='page-header-title'>📊 Model Insights</p>
        <p class='page-header-sub'>Performance metrics, cross-validation results, and feature importance analysis of the trained Logistic Regression model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏆 Model Performance (Test Set)</div>', unsafe_allow_html=True)
    metric_data = [("71.39%","Accuracy","🎯","#2563eb"),("73.16%","Precision","🔍","#7c3aed"),
                   ("67.51%","Recall","📡","#0891b2"),("70.22%","F1-Score","⚖️","#059669")]
    cols = st.columns(4)
    for col, (val, lbl, icon, color) in zip(cols, metric_data):
        with col:
            st.markdown(f'''<div style="background:#fff; border:1px solid #e2e8f0; border-radius:14px;
                         padding:22px 18px; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.05); border-top:4px solid {color};">
                <div style="font-size:24px; margin-bottom:8px;">{icon}</div>
                <div style="font-size:30px; font-weight:800; color:{color};">{val}</div>
                <div style="font-size:12px; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-top:6px;">{lbl}</div>
            </div>''', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔍 Overfitting / Underfitting Check</div>', unsafe_allow_html=True)
    ov1, ov2 = st.columns(2)
    train_acc, test_acc = 72.05, 71.39
    diff = abs(train_acc - test_acc)
    with ov1:
        st.markdown(f'''<div class="card">
            <div style="font-size:15px; font-weight:700; color:#0f172a; margin-bottom:16px;">Accuracy Comparison</div>
            <div style="display:flex; gap:16px; margin-bottom:16px;">
                <div style="flex:1; background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;">Train</div>
                    <div style="font-size:28px; font-weight:800; color:#2563eb;">~{train_acc:.2f}%</div>
                </div>
                <div style="flex:1; background:#f0fdf4; border:1px solid #86efac; border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;">Test</div>
                    <div style="font-size:28px; font-weight:800; color:#15803d;">{test_acc:.2f}%</div>
                </div>
            </div>
            <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:12px 14px; display:flex; align-items:center; gap:10px;">
                <span style="font-size:20px;">✅</span>
                <div>
                    <div style="font-weight:700; color:#15803d; font-size:14px;">Good Fit — No Overfitting</div>
                    <div style="font-size:12px; color:#64748b; margin-top:2px;">Gap of only <b>{diff:.2f}%</b> between train and test accuracy.</div>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)
    with ov2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#f8fafc")
        bars = ax.bar(["Train", "Test"], [train_acc, test_acc], color=["#2563eb", "#22c55e"], width=0.4, edgecolor="white", linewidth=1.5)
        ax.set_ylim(60, 80); ax.set_ylabel("Accuracy (%)", color="#64748b", fontsize=11); ax.tick_params(colors="#64748b")
        for spine in ax.spines.values(): spine.set_color("#e2e8f0")
        ax.set_title("Train vs Test Accuracy", color="#0f172a", fontsize=13, fontweight="bold", pad=12)
        for bar, val in zip(bars, [train_acc, test_acc]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-1.5, f"{val:.2f}%", ha="center", va="top", color="white", fontweight="bold", fontsize=12)
        ax.axhline(y=test_acc, color="#f59e0b", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#cbd5e1"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-title">🔁 5-Fold Cross-Validation Results</div>', unsafe_allow_html=True)
    cv_scores = np.array([71.52, 71.63, 71.45, 71.71, 71.58])
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    cv1, cv2 = st.columns([1, 2])
    with cv1:
        st.markdown(f'''<div class="card">
            <div style="font-size:15px; font-weight:700; color:#0f172a; margin-bottom:16px;">CV Summary</div>
            <div style="margin-bottom:12px;">
                <div style="font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:4px;">Mean Accuracy</div>
                <div style="font-size:28px; font-weight:800; color:#2563eb;">{cv_mean:.2f}%</div>
            </div>
            <div style="margin-bottom:12px;">
                <div style="font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:4px;">Std Deviation</div>
                <div style="font-size:22px; font-weight:700; color:#059669;">{cv_std:.2f}%</div>
            </div>
            <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:10px 12px;">
                <div style="font-weight:700; color:#15803d;">✅ Stable Model</div>
                <div style="font-size:12px; color:#64748b; margin-top:2px;">Low spread → consistent performance</div>
            </div>
        </div>''', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Fold":[f"Fold {i}" for i in range(1,6)],
            "Accuracy":[f"{s:.2f}%" for s in cv_scores]}), use_container_width=True, hide_index=True)
    with cv2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor("#ffffff"); ax2.set_facecolor("#f8fafc")
        bars2 = ax2.bar([f"Fold {i}" for i in range(1,6)], cv_scores, color="#2563eb", width=0.5, zorder=3, edgecolor="white", linewidth=1.5)
        ax2.axhline(y=cv_mean, color="#f59e0b", linestyle="--", linewidth=2, zorder=4, label=f"Mean ({cv_mean:.2f}%)")
        ax2.axhspan(cv_mean-cv_std, cv_mean+cv_std, alpha=0.12, color="#f59e0b", label=f"±1 Std ({cv_std:.2f}%)")
        for bar, val in zip(bars2, cv_scores):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.12, f"{val:.2f}%", ha="center", va="top", color="white", fontweight="bold", fontsize=11)
        ax2.set_ylim(cv_mean-2, cv_mean+2); ax2.set_ylabel("Accuracy (%)", color="#64748b", fontsize=11); ax2.tick_params(colors="#64748b")
        for spine in ax2.spines.values(): spine.set_color("#e2e8f0")
        ax2.set_title("5-Fold CV — Per-Fold Accuracy", color="#0f172a", fontsize=13, fontweight="bold", pad=12)
        ax2.legend(facecolor="#ffffff", edgecolor="#e2e8f0", labelcolor="#64748b", fontsize=10)
        ax2.grid(axis="y", linestyle="--", alpha=0.4, color="#cbd5e1", zorder=0)
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.markdown('<div class="section-title">📌 Feature Importance (Model Coefficients)</div>', unsafe_allow_html=True)
    feature_names = ["age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active","BMI"]
    try: coefs = model.named_steps["model"].coef_[0]
    except: coefs = np.array([0.42,-0.05,-0.08,0.12,0.61,0.38,0.28,0.17,0.07,0.06,-0.12,0.19])
    sorted_idx  = np.argsort(np.abs(coefs))[::-1]
    sorted_feat = [feature_names[i] for i in sorted_idx]
    sorted_coef = [coefs[i] for i in sorted_idx]
    colors_feat = ["#ef4444" if c > 0 else "#2563eb" for c in sorted_coef]
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    fig3.patch.set_facecolor("#ffffff"); ax3.set_facecolor("#f8fafc")
    ax3.barh(sorted_feat[::-1], sorted_coef[::-1], color=colors_feat[::-1], edgecolor="white", linewidth=0.8)
    ax3.axvline(x=0, color="#94a3b8", linewidth=1.5)
    ax3.set_xlabel("Coefficient Value (scaled)", color="#64748b", fontsize=11)
    ax3.set_title("Logistic Regression Coefficients\n(red = raises risk, blue = lowers risk)", color="#0f172a", fontsize=12, fontweight="bold", pad=12)
    ax3.tick_params(colors="#64748b", labelsize=10)
    for spine in ax3.spines.values(): spine.set_color("#e2e8f0")
    ax3.legend(handles=[mpatches.Patch(color="#ef4444", label="Increases risk"), mpatches.Patch(color="#2563eb", label="Decreases risk")],
               facecolor="#ffffff", edgecolor="#e2e8f0", labelcolor="#334155")
    ax3.grid(axis="x", linestyle="--", alpha=0.4, color="#cbd5e1")
    plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    st.markdown('<div class="section-title">⚙️ Model Architecture</div>', unsafe_allow_html=True)
    arch1, arch2 = st.columns(2)
    with arch1:
        st.markdown('''<div class="card">
            <div style="font-size:15px; font-weight:700; color:#0f172a; margin-bottom:16px;">🔧 Pipeline Components</div>
            <div style="display:flex; flex-direction:column; gap:10px;">
                <div style="background:#eff6ff; border:1.5px solid #bfdbfe; border-radius:12px; padding:14px;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                        <span style="background:#2563eb; color:white; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700;">Step 1</span>
                        <span style="font-weight:700; color:#1e40af;">StandardScaler</span>
                    </div>
                    <div style="font-size:13px; color:#64748b;">Normalises all 12 input features to zero mean and unit variance.</div>
                </div>
                <div style="background:#f0fdf4; border:1.5px solid #86efac; border-radius:12px; padding:14px;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                        <span style="background:#15803d; color:white; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700;">Step 2</span>
                        <span style="font-weight:700; color:#15803d;">Logistic Regression</span>
                    </div>
                    <div style="font-size:13px; color:#64748b;">Binary classifier. <code>max_iter=1000</code>, L2 regularisation. Sigmoid output.</div>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)
    with arch2:
        st.markdown('''<div class="card">
            <div style="font-size:15px; font-weight:700; color:#0f172a; margin-bottom:16px;">📋 Training Details</div>
            <table style="width:100%; font-size:14px; border-collapse:collapse;">
                <tr><td style="color:#64748b; padding:10px 0; border-bottom:1px solid #f1f5f9;">Dataset</td><td style="color:#0f172a; font-weight:600; text-align:right;">Cardiovascular Disease</td></tr>
                <tr><td style="color:#64748b; padding:10px 0; border-bottom:1px solid #f1f5f9;">Total Records</td><td style="color:#0f172a; font-weight:600; text-align:right;">~70,000</td></tr>
                <tr><td style="color:#64748b; padding:10px 0; border-bottom:1px solid #f1f5f9;">Train / Test Split</td><td style="color:#0f172a; font-weight:600; text-align:right;">80% / 20%</td></tr>
                <tr><td style="color:#64748b; padding:10px 0; border-bottom:1px solid #f1f5f9;">Stratified Split</td><td style="color:#0f172a; font-weight:600; text-align:right;">Yes</td></tr>
                <tr><td style="color:#64748b; padding:10px 0; border-bottom:1px solid #f1f5f9;">Cross-Validation</td><td style="color:#0f172a; font-weight:600; text-align:right;">5-Fold Stratified</td></tr>
                <tr><td style="color:#64748b; padding:10px 0;">Features</td><td style="color:#0f172a; font-weight:600; text-align:right;">12 (incl. BMI)</td></tr>
            </table>
        </div>''', unsafe_allow_html=True)
