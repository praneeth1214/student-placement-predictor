import pandas as pd

def prepare_input(cgpa, attendance, projects, internships, skills, backlogs):
    return pd.DataFrame(
        [[cgpa, attendance, projects, internships, skills, backlogs]],
        columns=[
            "cgpa",
            "attendance",
            "projects",
            "internships",
            "skills",
            "backlogs"
        ]
    )
