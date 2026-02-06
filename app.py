import streamlit as st
import joblib
import os
import pandas as pd
from src.utils import prepare_input

# --------------------------------------------------
# Load model safely
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Student Placement Predictor",
    layout="centered"
)

st.title("🎓 Student Placement Predictor")
st.caption(
    "Predict placement probability and receive clear, actionable improvement guidance."
)

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("📥 Student Profile")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
skills = st.slider("Skill Level (0–10)", 0, 10, 6)
projects = st.number_input("Projects Completed", 0, 10, 2)

attendance = st.slider("Attendance (%)", 0, 100, 75)
internships = st.selectbox("Internships Done", [0, 1])
backlogs = st.selectbox("Any Backlogs?", ["No", "Yes"])
backlogs_val = 1 if backlogs == "Yes" else 0

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Placement", use_container_width=True):

    input_df = prepare_input(
        cgpa=cgpa,
        attendance=attendance,
        projects=projects,
        internships=internships,
        skills=skills,
        backlogs=backlogs_val
    )

    probability = model.predict_proba(input_df)[0][1]

    # ---------------- Result ----------------
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Placement Probability", f"{probability * 100:.1f}%")

    with col2:
        if probability >= 0.7:
            st.success("High Chance of Placement")
        elif probability >= 0.4:
            st.warning("Moderate Chance of Placement")
        else:
            st.error("Low Chance of Placement")

    # ---------------- Uncertainty ----------------
    if 0.45 <= probability <= 0.55:
        st.info(
            "⚠️ Prediction uncertainty is high. Small improvements "
            "in CGPA, skills, or projects may change the outcome."
        )

    st.divider()

    # ---------------- Eligibility Check ----------------
    st.subheader("🚫 Eligibility Check")

    if backlogs_val == 1:
        st.error(
            "Backlogs act as a strong eligibility filter and "
            "significantly reduce placement chances."
        )
    else:
        st.success(
            "No backlogs detected — eligible for ranking evaluation."
        )

    st.divider()

    # ---------------- Suggestions ----------------
    st.subheader("📌 Personalized Improvement Suggestions")

    suggestions = []

    if backlogs_val == 1:
        suggestions.append("🔴 Clear backlogs — this is the biggest blocker.")
    if cgpa < 7:
        suggestions.append("🟠 Improve CGPA to at least 7.0.")
    if skills < 6:
        suggestions.append("🟡 Strengthen technical skills with focused practice.")
    if projects < 2:
        suggestions.append("🔵 Build more real-world projects.")
    if internships == 0:
        suggestions.append("🟣 Gain internship or industry exposure.")

    if not suggestions:
        suggestions.append("🟢 Strong profile — focus on interview preparation.")

    for s in suggestions:
        st.write(s)

    st.divider()

    # ---------------- Feature Importance (SAFE) ----------------
    st.subheader("🔍 Key Ranking Factors")

    # SAFELY extract coefficients
    if hasattr(model, "base_estimator"):
        # CalibratedClassifierCV
        coef = model.base_estimator.named_steps["model"].coef_[0]
    else:
        # Plain Pipeline
        coef = model.named_steps["model"].coef_[0]

    features = input_df.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Impact": [abs(c) for c in coef]
    }).sort_values(by="Impact", ascending=False)

    # Exclude backlogs from ranking chart
    ranking_df = importance_df[importance_df["Feature"] != "backlogs"]

    st.bar_chart(
        ranking_df.set_index("Feature"),
        height=250
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
with st.expander("ℹ️ About This Predictor"):
    st.write(
        """
        **Model:** Logistic Regression (with scaling and calibration)  
        **Evaluation:** Cross-validation, ROC-AUC, uncertainty handling  

        **Interpretation Logic:**
        - Backlogs → eligibility gate  
        - CGPA, skills, projects → ranking factors  

        **Disclaimer:** This tool provides guidance, not guarantees.
        """
    )
