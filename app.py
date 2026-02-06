import streamlit as st
import joblib
import os
from src.utils import prepare_input

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Student Placement Predictor")
st.title("🎓 Student Placement Predictor")

# Inputs
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
attendance = st.slider("Attendance (%)", 0, 100, 75)
projects = st.number_input("Projects Completed", 0, 10, 2)
internships = st.selectbox("Internships Done", [0, 1])
skills = st.slider("Skill Rating (0–10)", 0, 10, 6)
backlogs = st.number_input("Backlogs", 0, 10, 0)

if st.button("Predict Placement"):
    input_df = prepare_input(
        cgpa, attendance, projects,
        internships, skills, backlogs
    )

    probability = model.predict_proba(input_df)[0][1]

    st.subheader(f"📈 Placement Probability: {probability * 100:.2f}%")

    # Risk category
    if probability >= 0.7:
        st.success("🟢 Low Risk – Strong placement chances")
    elif probability >= 0.4:
        st.warning("🟡 Medium Risk – Scope for improvement")
    else:
        st.error("🔴 High Risk – Needs focused improvement")

    # Priority explanation
    st.subheader("🔍 Prediction Priority")
    st.write("""
    The prediction is primarily influenced in the following order:
    1. **CGPA** – academic consistency (highest impact)
    2. **Skills** – job readiness
    3. **Projects** – practical exposure
    """)

    # Actionable guidance aligned with priority
    st.subheader("📌 Recommended Focus Path")

    if cgpa < 7:
        st.write("🔴 **Primary focus:** Improve CGPA (strongest influence).")
    elif skills < 6:
        st.write("🟠 **Secondary focus:** Improve technical skills.")
    elif projects < 2:
        st.write("🟡 **Tertiary focus:** Build more real-world projects.")
    else:
        st.write("🟢 Strong profile across key influencing factors.")

    # Feature importance (transparent, not fake)
    st.subheader("📊 Key Influencing Factors (Model-Based)")
    coef = model.named_steps["model"].coef_[0]
    features = input_df.columns

    importance = sorted(
        zip(features, coef),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for feature, value in importance[:3]:
        st.write(f"- **{feature}**")

# Footer
with st.expander("ℹ️ Model Information"):
    st.write("""
    - Model: Logistic Regression (with feature scaling)
    - Output: Probability of placement
    - Note: Predictions are advisory, not deterministic decisions
    """)
