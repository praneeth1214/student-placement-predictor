import streamlit as st
import joblib
import os
from src.utils import prepare_input

# --------------------------------------------------
# Load model
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Session state (CRITICAL for What-If)
# --------------------------------------------------
if "base_probability" not in st.session_state:
    st.session_state.base_probability = None

if "base_input" not in st.session_state:
    st.session_state.base_input = None

# --------------------------------------------------
# Role → Skills mapping (8 professions)
# --------------------------------------------------
ROLE_SKILLS = {
    "Web Developer": ["HTML/CSS", "JavaScript", "React", "Backend Basics", "Git"],
    "Backend Developer": ["Python / Java", "Databases", "APIs", "System Design", "Git"],
    "Full Stack Developer": ["HTML/CSS", "JavaScript", "Frontend Framework", "Backend", "Databases"],
    "Data Analyst": ["Python", "SQL", "Excel", "Statistics", "Data Visualization"],
    "Data Scientist": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas / NumPy"],
    "Machine Learning Engineer": ["Python", "Machine Learning", "Deep Learning", "Model Deployment", "Data Handling"],
    "AI Engineer": ["Python", "Deep Learning", "Neural Networks", "AI Frameworks", "Math for AI"],
    "Software Engineer": ["Data Structures", "Algorithms", "Programming", "OOP", "Git"]
}

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Career-Aware Placement Predictor",
    layout="centered"
)

st.title("🎓 Career-Aware Placement Predictor")
st.caption("Placement prediction using academics, experience, and role-specific skills.")

st.divider()

# --------------------------------------------------
# Step 1: Profession
# --------------------------------------------------
st.subheader("🎯 Target Profession")
role = st.selectbox(
    "Select the role you are preparing for",
    list(ROLE_SKILLS.keys())
)

st.divider()

# --------------------------------------------------
# Step 2: Skills (role-dependent)
# --------------------------------------------------
st.subheader("🧠 Skills You Know")

selected_skills = []
for skill in ROLE_SKILLS[role]:
    if st.checkbox(skill):
        selected_skills.append(skill)

total_skills = len(ROLE_SKILLS[role])
skill_score = (len(selected_skills) / total_skills) * 10

st.caption(f"Calculated Skill Score: **{skill_score:.1f} / 10**")

st.divider()

# --------------------------------------------------
# Step 3: Academic & Experience Details
# --------------------------------------------------
st.subheader("📘 Academic & Experience Details")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
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
        skills=skill_score,
        backlogs=backlogs_val
    )

    probability = model.predict_proba(input_df)[0][1]

    # Save base state for What-If
    st.session_state.base_probability = probability
    st.session_state.base_input = {
        "cgpa": cgpa,
        "attendance": attendance,
        "projects": projects,
        "internships": internships,
        "skills": skill_score,
        "backlogs": backlogs_val
    }

    st.subheader("📊 Prediction Result")
    st.metric("Placement Probability", f"{probability * 100:.1f}%")

    if probability >= 0.7:
        st.success("High Chance of Placement")
    elif probability >= 0.4:
        st.warning("Moderate Chance of Placement")
    else:
        st.error("Low Chance of Placement")

    st.info(
        "Prediction is based on the **combined effect** of role-specific skills, "
        "CGPA, projects, internships, attendance, and backlogs."
    )

    st.divider()

# --------------------------------------------------
# WHAT-IF Skill Simulator (FIXED)
# --------------------------------------------------
st.subheader("🧪 What-If Skill Simulator")

if st.session_state.base_probability is None:
    st.info("Run a placement prediction first to enable simulation.")
else:
    missing_skills = list(set(ROLE_SKILLS[role]) - set(selected_skills))

    if len(missing_skills) == 0:
        st.success("You already meet all skill requirements for this role.")
    else:
        skill_to_simulate = st.selectbox(
            "Select a skill to simulate learning",
            missing_skills,
            key="whatif_skill"
        )

        if st.button("Simulate Skill Improvement", key="simulate_btn"):
            simulated_skill_count = len(selected_skills) + 1
            simulated_skill_score = (simulated_skill_count / total_skills) * 10

            simulated_input = prepare_input(
                cgpa=st.session_state.base_input["cgpa"],
                attendance=st.session_state.base_input["attendance"],
                projects=st.session_state.base_input["projects"],
                internships=st.session_state.base_input["internships"],
                skills=simulated_skill_score,
                backlogs=st.session_state.base_input["backlogs"]
            )

            simulated_probability = model.predict_proba(simulated_input)[0][1]
            delta = (simulated_probability - st.session_state.base_probability) * 100

            st.metric(
                "New Placement Probability",
                f"{simulated_probability * 100:.1f}%",
                delta=f"{delta:+.1f}%"
            )

            st.caption(
                f"Learning **{skill_to_simulate}** increases skill score "
                f"from {st.session_state.base_input['skills']:.1f} → {simulated_skill_score:.1f}."
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------
with st.expander("ℹ️ About This System"):
    st.write(
        """
        - Profession-aware skill evaluation  
        - Skills are **computed**, not self-rated  
        - Backlogs are a **penalty**, not a hard rule  
        - WHAT-IF simulator shows **decision impact**, not guarantees  
        - Designed to reflect real hiring trade-offs
        """
    )
