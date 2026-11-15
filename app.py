import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import engineer, load_clean

st.set_page_config(page_title="Signal | Churn Predictor", page_icon="📡", layout="wide")

# ---------- THEME ----------
INK, PANEL, LINE = "#F7FBFF", "#FFFFFF", "#DCE8F5"
GLOW, HERO2 = "#5EEAD4", "#00C2B8"
ACCENT, ACCENT2 = "#3B82F6", "#14B8A6"
TEXT, MUTED = "#14284B", "#5B6F8F"
SAFE, WARN, DANGER = "#10B981", "#F59E0B", "#F43F5E"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, label {{ font-family: 'DM Sans', sans-serif !important; }}
h1, h2, h3, .brand {{ font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.02em; }}
.stApp {{ background: radial-gradient(at 8% 12%, #F4A58A 0, transparent 45%), radial-gradient(at 92% 8%, #7FAEF5 0, transparent 48%), radial-gradient(at 90% 92%, #A18CF0 0, transparent 48%), radial-gradient(at 8% 92%, #6FD3C0 0, transparent 45%), #7D9BCB; background-attachment: fixed; color: {TEXT}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0F1B45, #1E2F6B); border-right: 1px solid #2A3C7E; }}
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: #FFFFFF; }}
.block-container {{ max-width: 1200px; margin-top: 1.5rem; padding: 2rem 2.5rem !important;
  background: rgba(255,255,255,.92); border-radius: 28px; box-shadow: 0 20px 60px rgba(15,27,69,.35); }}
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp [data-testid="stWidgetLabel"] p {{ color: {TEXT}; }}
.hero {{ padding: 1.6rem 2rem; border-radius: 22px; margin-bottom: 1.3rem;
  background: linear-gradient(120deg, #0F1B45 0%, #25409E 55%, {ACCENT2} 100%); box-shadow: 0 14px 34px rgba(59,130,246,.35); }}
.stApp .hero h1 {{ margin: 0; font-size: 2.3rem; font-weight: 800; color: #FFFFFF; text-shadow: 0 2px 8px rgba(10,30,90,.35); }}
.stApp .hero p {{ margin: .3rem 0 0; color: rgba(255,255,255,.95); text-shadow: 0 1px 4px rgba(10,30,90,.3); }}
.brand {{ font-size: 1.4rem; font-weight: 800; padding: .2rem 0 1rem; color: #FFFFFF; }}
.card {{ background: linear-gradient(90deg, #FF8A6B, {ACCENT}, {ACCENT2}) top/100% 5px no-repeat, {PANEL}; border: 1px solid {LINE}; border-radius: 18px; padding: 1.1rem 1.3rem;
  box-shadow: 0 8px 24px rgba(40,80,150,.15); }}
.pill {{ display:inline-block; padding: .25rem .9rem; border-radius: 999px; font-weight: 700; font-size: .95rem; }}
.tip {{ border-left: 4px solid {ACCENT2}; padding: .55rem .9rem; margin: .5rem 0; background: #F1FAF8; border-radius: 0 12px 12px 0; }}
.stButton > button {{ background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); color: #FFFFFF; font-weight: 800;
  border: 0; border-radius: 14px; padding: .7rem 2.2rem; font-size: 1.05rem; box-shadow: 0 8px 20px rgba(59,130,246,.35); }}
.stButton > button:hover {{ filter: brightness(1.08); color: #FFFFFF; }}
.stButton button p {{ color: #FFFFFF; }}
[data-testid="stTabs"] button {{ font-weight: 700; }}
[data-testid="stImage"] img {{ border-radius: 16px; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
                     "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": LINE, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False})
CH = {"No": SAFE, "Yes": DANGER}

MODEL_FILES = {"Best Model (Tuned)": "models/best_model.joblib",
               "Logistic Regression": "models/logistic_regression.joblib",
               "KNN": "models/knn.joblib",
               "Random Forest": "models/random_forest.joblib",
               "Gradient Boosting": "models/gradient_boosting.joblib"}

@st.cache_resource
def get_model(path): return joblib.load(path)

@st.cache_data
def get_data(): return load_clean()

def gauge(p):
    color = SAFE if p < .35 else WARN if p < .6 else DANGER
    off = 251.3 * (1 - p)
    return (f'<div class="card" style="text-align:center">'
            f'<svg viewBox="0 0 200 120" width="100%" style="max-width:360px">'
            f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{LINE}" stroke-width="16" stroke-linecap="round"/>'
            f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" '
            f'stroke-dasharray="251.3" stroke-dashoffset="{off}"/>'
            f'<text x="100" y="88" text-anchor="middle" font-size="34" font-weight="800" fill="{TEXT}" '
            f'font-family="Bricolage Grotesque, sans-serif">{p:.0%}</text>'
            f'<text x="100" y="108" text-anchor="middle" font-size="9" fill="{MUTED}">chance of leaving</text></svg></div>')

st.sidebar.markdown('<div class="brand">📡 Signal</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Predict", "Explore data", "Model results", "About"], label_visibility="collapsed")

# ---------------- PREDICT ----------------
if page == "Predict":
    st.markdown('<div class="hero"><h1>Will this customer stay?</h1>'
                '<p>Describe the customer, and the model estimates how likely they are to leave.</p></div>',
                unsafe_allow_html=True)
    choice = st.sidebar.selectbox("Model", list(MODEL_FILES))
    model = get_model(MODEL_FILES[choice])

    YN = ["Yes", "No"]; YNI = ["Yes", "No", "No internet service"]
    t1, t2, t3 = st.tabs(["👤 Customer", "📶 Services", "💳 Billing"])
    with t1:
        a, b = st.columns(2)
        gender = a.selectbox("Gender", ["Male", "Female"])
        senior = b.selectbox("Senior citizen", ["No", "Yes"])
        partner = a.selectbox("Has partner", YN)
        dependents = b.selectbox("Has dependents", YN)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with t2:
        a, b = st.columns(2)
        phone = a.selectbox("Phone service", YN)
        multi = b.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        internet = a.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        security = b.selectbox("Online security", YNI)
        backup = a.selectbox("Online backup", YNI)
        protection = b.selectbox("Device protection", YNI)
        support = a.selectbox("Tech support", YNI)
        tv = b.selectbox("Streaming TV", YNI)
        movies = a.selectbox("Streaming movies", YNI)
    with t3:
        a, b = st.columns(2)
        contract = a.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = b.selectbox("Paperless billing", YN)
        payment = a.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = b.number_input("Monthly charges ($)", 18.0, 120.0, 70.0)

    if st.button("Check churn risk"):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": monthly * tenure}])
        proba = float(model.predict_proba(engineer(row))[0, 1])
        level, col = (("High risk", DANGER) if proba >= .6 else ("Medium risk", WARN) if proba >= .35
                      else ("Low risk", SAFE))

        st.write("")
        g, r = st.columns([1, 1.3])
        g.markdown(gauge(proba), unsafe_allow_html=True)
        verdict = "likely to leave" if proba >= .5 else "likely to stay"
        r.markdown(f'<div class="card"><span class="pill" style="background:{col}22;color:{col}">{level}</span>'
                   f'<h2 style="margin:.6rem 0 .2rem">This customer is {verdict}.</h2>'
                   f'<p style="color:{MUTED}">Estimated by {choice}.</p></div>', unsafe_allow_html=True)

        tips = []
        if contract == "Month-to-month": tips.append("Offer a discount for a 1- or 2-year contract.")
        if tenure < 12: tips.append("New customer: start an onboarding or loyalty programme.")
        if internet == "Fiber optic" and support == "No": tips.append("Offer free tech support to fiber users.")
        if payment == "Electronic check": tips.append("Encourage automatic payment methods.")
        st.subheader("What you can do")
        for t in tips or ["No special action needed."]:
            st.markdown(f'<div class="tip">{t}</div>', unsafe_allow_html=True)

        st.subheader("What the model looks at most")
        fi = pd.read_csv("models/feature_importance.csv").head(8)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=fi, y="feature", x="importance", ax=ax, color=ACCENT)
        ax.set_ylabel("")
        st.pyplot(fig, use_container_width=False)

# ---------------- DATA ----------------
elif page == "Explore data":
    st.markdown('<div class="hero"><h1>Explore the data</h1>'
                '<p>What separates customers who leave from those who stay.</p></div>', unsafe_allow_html=True)
    df = get_data()
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(); sns.countplot(data=df, x="Churn", hue="Churn", palette=CH, legend=False, ax=ax)
        ax.set_title("Churn distribution"); st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(); sns.countplot(data=df, x="Contract", hue="Churn", palette=CH, ax=ax)
        ax.set_title("Churn by contract"); st.pyplot(fig)
    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(); sns.histplot(data=df, x="tenure", hue="Churn", palette=CH, bins=30, ax=ax)
        ax.set_title("Tenure vs churn"); st.pyplot(fig)
    with c4:
        fig, ax = plt.subplots(); sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn",
                                              palette=CH, legend=False, ax=ax)
        ax.set_title("Monthly charges vs churn"); st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="GnBu", ax=ax)
    ax.set_title("Correlation"); st.pyplot(fig, use_container_width=False)

# ---------------- RESULTS ----------------
elif page == "Model results":
    st.markdown('<div class="hero"><h1>How well do the models work?</h1>'
                '<p>Four models compared, then the best one tuned.</p></div>', unsafe_allow_html=True)
    meta = json.load(open("models/meta.json"))
    st.markdown(f'<div class="card"><b>Best model:</b> {meta["best_model"]}<br>'
                f'<span style="color:{MUTED}">Tuned parameters: {meta["best_params"]}</span></div>',
                unsafe_allow_html=True)
    st.write("")
    st.dataframe(pd.read_csv("models/results.csv").round(4), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image("assets/confusion_matrix.png", caption="Confusion matrix")
    c2.image("assets/roc_curves.png", caption="ROC curves")

else:
    st.markdown('<div class="hero"><h1>About</h1>'
                '<p>An end-to-end ML app: data cleaning, four models, tuning and deployment.</p></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card">Dataset: IBM Telco Customer Churn.<br>'
                'Built with Python, Scikit-learn and Streamlit.</div>', unsafe_allow_html=True)