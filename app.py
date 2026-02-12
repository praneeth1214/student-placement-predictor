import streamlit as st
import joblib
import os
import plotly.graph_objects as go
from src.utils import prepare_input

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Career Intelligence Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# ENTERPRISE UI STYLE
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0b1120;
}
h1, h2, h3 {
    color: #e2e8f0;
}
.section-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    height: 45px;
    border: none;
}
.stButton>button:hover {
    background-color: #1d4ed8;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# ROLE SKILL MAPPING
# --------------------------------------------------
ROLE_SKILLS = {
    "Web Developer": ["HTML/CSS", "JavaScript", "React", "Backend Basics", "Git"],
    "Data Analyst": ["Python", "SQL", "Excel", "Statistics", "Data Visualization"],
    "Machine Learning Engineer": ["Python", "Machine Learning", "Deep Learning", "Model Deployment", "Pandas"],
    "AI Engineer": ["Python", "Deep Learning", "Neural Networks", "AI Frameworks", "Math for AI"],
    "Software Engineer": ["Data Structures", "Algorithms", "Programming", "OOP", "Git"]
}

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1>🧠 AI Career Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Enterprise-grade student career readiness and placement analytics")
st.divider()

# --------------------------------------------------
# ROLE SECTION
# --------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("🎯 Target Role Selection")

role = st.selectbox("Select Role", list(ROLE_SKILLS.keys()))
required_skills = ROLE_SKILLS[role]
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# SKILL SECTION
# --------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("🧠 Skill Assessment Matrix")

selected_skills = []
cols = st.columns(3)

for i, skill in enumerate(required_skills):
    with cols[i % 3]:
        if st.checkbox(skill):
            selected_skills.append(skill)

skill_score = (len(selected_skills) / len(required_skills)) * 10
missing_skills = list(set(required_skills) - set(selected_skills))

fig = go.Figure()
fig.add_trace(go.Bar(
    x=required_skills,
    y=[1 if s in selected_skills else 0 for s in required_skills],
    marker_color="#2563eb"
))

fig.update_layout(
    height=300,
    plot_bgcolor="#111827",
    paper_bgcolor="#111827",
    font_color="#e2e8f0",
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig, use_container_width=True)

if missing_skills:
    st.warning("Skill Gaps Identified")
else:
    st.success("All Required Skills Covered")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ACADEMIC SECTION
# --------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📘 Academic & Experience Profile")

col1, col2, col3 = st.columns(3)

with col1:
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)

with col2:
    projects = st.number_input("Projects", 0, 10, 2)

with col3:
    internships = st.selectbox("Internships", [0, 1, 2])

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# READINESS SCORE
# --------------------------------------------------
readiness_score = (
    (cgpa / 10) * 40 +
    (skill_score / 10) * 35 +
    (projects / 10) * 15 +
    (internships * 5)
)

readiness_score = max(0, min(100, readiness_score))

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("🚀 Career Readiness Index")

st.progress(readiness_score / 100)
st.metric("Overall Readiness", f"{readiness_score:.1f}%")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📊 Placement Prediction Engine")

if st.button("Analyze Placement Potential", use_container_width=True):

    input_df = prepare_input(
        cgpa=cgpa,
        attendance=75,   # kept for model compatibility
        projects=projects,
        internships=internships,
        skills=skill_score,
        backlogs=0
    )

    probability = model.predict_proba(input_df)[0][1]

    colA, colB = st.columns(2)

    with colA:
        st.metric("Placement Probability", f"{probability*100:.1f}%")

    with colB:
        st.metric("Readiness Score", f"{readiness_score:.1f}%")

    if probability >= 0.75:
        st.success("High Placement Stability")
    elif probability >= 0.6:
        st.warning("Moderate Stability")
    else:
        st.error("Low Stability – Improvement Required")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("<center style='color:#64748b;'>AI Career Intelligence • Enterprise Edition </center>", unsafe_allow_html=True)
