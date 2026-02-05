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

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
attendance = st.slider("Attendance (%)", 0, 100, 75)
projects = st.number_input("Projects Completed", 0, 10, 2)
internships = st.selectbox("Internships Done", [0, 1])
skills = st.slider("Skill Rating (0–10)", 0, 10, 6)
backlogs = st.number_input("Backlogs", 0, 10, 0)

if st.button("Predict Placement"):
    input_df = prepare_input(
        cgpa,
        attendance,
        projects,
        internships,
        skills,
        backlogs
    )

    probability = model.predict_proba(input_df)[0][1]

    st.subheader(f"Placement Probability: {probability * 100:.2f}%")

    if probability >= 0.7:
        st.success("Low Risk – Good placement chances")
    elif probability >= 0.4:
        st.warning("Medium Risk – Improve skills/projects")
    else:
        st.error("High Risk – Immediate improvement needed")
