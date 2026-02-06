import streamlit as st
import joblib
import os
import pandas as pd
from src.utils import prepare_input

# --------------------------------------------------
# Load trained model (Pipeline: StandardScaler + LogisticRegression)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Student Placement Predictor",
    layout="centered"
)

st.title("🎓 Student Placement Predictor")
st.write(
    "This application predicts **placement probability** based on academic and "
    "skill-related factors. Predictions are **advisory**, not deterministic."
)

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.subheader("📥 Enter Student Details")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
skills = st.slider("Skill Rating (0–10)", 0, 10, 6)
projects = st.number_input("Number of Projects", min_value=0, max_value=10, value=2)

attendance = st.slider("Attendance (%)", 0, 100, 75)
internships = st.selectbox("Internships Done", [0, 1])
backlogs = st.selectbox("Has Backlogs?", ["No", "Yes"])
backlogs_val = 1 if backlogs == "Yes" else 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Placement"):
    input_df = prepare_input(
        cgpa=cgpa,
        attendance=attendance,
        projects=projects,
        internships=internships,
        skills=skills,
        backlogs=backlogs_val
    )

    probability = model.predict_proba(input_df)[0][1]

    st.subheader(f"📈 Placement Probability: **{probability * 100:.2f}%**")

    # Risk Category
    if probability >= 0.7:
        st.success("🟢 Low Risk – Strong placement chances")
    elif probability >= 0.4:
        st.warning("🟡 Medium Risk – Needs improvement")
    else:
        st.error("🔴 High Risk – Low placement probability")

    # --------------------------------------------------
    # Eligibility Gate (Backlogs)
    # --------------------------------------------------
    st.subheader("🚫 Eligibility Risk Check")

    if backlogs_val == 1:
        st.error(
            "Backlogs act as a **strong eligibility filter** in the model. "
            "Even with good CGPA or skills, backlogs significantly reduce "
            "placement probability."
        )
    else:
        st.success(
            "No backlogs detected — eligible for ranking based on CGPA, skills, "
            "and project experience."
        )

    # --------------------------------------------------
    # Model-Based Feature Impact
    # --------------------------------------------------
    coef = model.named_steps["model"].coef_[0]
    features = input_df.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Impact": coef,
        "Magnitude": [abs(c) for c in coef]
    }).sort_values(by="Magnitude", ascending=False)

    # --------------------------------------------------
    # Ranking Factors (Exclude Backlogs)
    # --------------------------------------------------
    st.subheader("📊 Ranking Factors (Among Eligible Students)")

    ranking_df = importance_df[importance_df["Feature"] != "backlogs"]

    st.bar_chart(
        ranking_df.set_index("Feature")[["Magnitude"]]
    )

    # --------------------------------------------------
    # Actionable Guidance (Aligned with Reality)
    # --------------------------------------------------
    st.subheader("📌 Recommended Focus Path")

    if backlogs_val == 1:
        st.write("🔴 **Top priority:** Clear backlogs to pass eligibility filters.")
    elif cgpa < 7:
        st.write("🟠 **Primary ranking factor:** Improve CGPA.")
    elif skills < 6:
        st.write("🟡 **Secondary ranking factor:** Improve technical skills.")
    elif projects < 2:
        st.write("🔵 **Supporting factor:** Build more real-world projects.")
    else:
        st.write("🟢 Strong profile across ranking factors.")

# --------------------------------------------------
# Model Info
# --------------------------------------------------
with st.expander("ℹ️ Model Information"):
    st.write(
        """
        - **Model:** Logistic Regression  
        - **Preprocessing:** StandardScaler  
        - **Interpretation:**  
          - Backlogs → Eligibility (risk gate)  
          - CGPA, Skills, Projects → Ranking factors  
        - **Note:** Predictions are probabilistic and advisory.
        """
    )
