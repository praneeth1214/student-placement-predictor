import streamlit as st
import joblib
import os
import pandas as pd
from src.utils import prepare_input

# --------------------------------------------------
# Load model
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
    "Placement prediction based on combined academic and skill profile."
)

st.divider()

# --------------------------------------------------
# Inputs
# --------------------------------------------------
st.subheader("📥 Student Profile")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
skills = st.slider("Skill Level (0–10)", 0, 10, 6)
projects = st.number_input("Projects Completed", 0, 10, 2)
internships = st.selectbox("Internships Done", [0, 1])
attendance = st.slider("Attendance (%)", 0, 100, 75)
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
            "Prediction is uncertain. Small improvements can significantly "
            "change the outcome."
        )

    st.divider()

    # ---------------- Explanation ----------------
    st.subheader("ℹ️ How This Prediction Works")

    st.write(
        "The prediction is based on the **combined effect** of CGPA, skills, "
        "projects, internships, attendance, and backlogs. "
        "**No single factor decides the outcome on its own.**"
    )

    if backlogs_val == 1:
        st.warning(
            "Backlogs reduce placement probability, but strong academics, skills, "
            "projects, and internships can offset this impact."
        )

    st.divider()

    # ---------------- Suggestions ----------------
    st.subheader("📌 Personalized Improvement Suggestions")

    suggestions = []

    if cgpa >= 7:
        suggestions.append("✔ CGPA is supporting your placement chances.")
    else:
        suggestions.append("⚠ Improving CGPA can significantly boost your chances.")

    if skills >= 6:
        suggestions.append("✔ Skill level positively impacts your profile.")
    else:
        suggestions.append("⚠ Strengthen technical skills to improve employability.")

    if projects >= 2:
        suggestions.append("✔ Projects add strong practical credibility.")
    else:
        suggestions.append("⚠ Build more real-world projects.")

    if internships == 1:
        suggestions.append("✔ Internship experience improves placement probability.")
    else:
        suggestions.append("⚠ Internship or industry exposure can help a lot.")

    if backlogs_val == 1:
        suggestions.append(
            "⚠ Clearing backlogs will further improve your chances, "
            "especially for top companies."
        )

    for s in suggestions:
        st.write(s)

    st.divider()

    # ---------------- Feature Insight ----------------
    st.subheader("🔍 Relative Influence of Factors")

    if hasattr(model, "base_estimator"):
        coef = model.base_estimator.named_steps["model"].coef_[0]
    else:
        coef = model.named_steps["model"].coef_[0]

    features = input_df.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Influence": [abs(c) for c in coef]
    }).sort_values(by="Influence", ascending=False)

    st.bar_chart(
        importance_df.set_index("Feature"),
        height=250
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
with st.expander("ℹ️ About This Model"):
    st.write(
        """
        **Model:** Logistic Regression (scaled & calibrated)  
        **Evaluation:** Cross-validation, ROC-AUC  

        **Design Principle:**  
        - Backlogs are treated as a **penalty**, not a hard rule  
        - Strong CGPA, skills, projects, and internships can offset negatives  

        **Disclaimer:** This is a guidance tool, not a guarantee.
        """
    )
