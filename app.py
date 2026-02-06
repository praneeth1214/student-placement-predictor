import streamlit as st
import joblib
import os
import pandas as pd
from src.utils import prepare_input

# --------------------------------------------------
# Load trained model (Pipeline: Scaler + LogisticRegression)
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
    "This application predicts **placement probability** based on "
    "academic and skill-related factors. "
    "Predictions are **advisory**, not deterministic."
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
backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=10, value=0)

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
        backlogs=backlogs
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
    # Priority Explanation
    # --------------------------------------------------
    st.subheader("🔍 Prediction Priority")
    st.write(
        """
        The model primarily considers features in the following order:
        1. **CGPA** – academic consistency (highest impact)
        2. **Skills** – job readiness
        3. **Projects** – practical exposure
        """
    )

    # --------------------------------------------------
    # Actionable Suggestions (Aligned with Priority)
    # --------------------------------------------------
    st.subheader("📌 Recommended Improvement Path")

    if cgpa < 7:
        st.write("🔴 **Primary focus:** Improve CGPA (highest influence on prediction).")
    elif skills < 6:
        st.write("🟠 **Secondary focus:** Strengthen technical skills.")
    elif projects < 2:
        st.write("🟡 **Tertiary focus:** Build more real-world projects.")
    else:
        st.write("🟢 Strong profile across key influencing factors.")

    # --------------------------------------------------
    # Feature Importance (Model-Based)
    # --------------------------------------------------
    st.subheader("📊 Feature Importance (Model-Based)")

    coef = model.named_steps["model"].coef_[0]
    features = input_df.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": [abs(c) for c in coef]
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(
        importance_df.set_index("Feature")
    )

# --------------------------------------------------
# Model Info Section
# --------------------------------------------------
with st.expander("ℹ️ Model Information"):
    st.write(
        """
        - **Model:** Logistic Regression  
        - **Preprocessing:** StandardScaler  
        - **Output:** Probability of placement  
        - **Note:** This tool is for guidance only, not final decisions.
        """
    )
