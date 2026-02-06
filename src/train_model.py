import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "students.csv")
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop("placed", axis=1)
y = df["placed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: scaling + logistic regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Inspect learned priority (for YOU, not UI)
coef = pipeline.named_steps["model"].coef_[0]
importance = sorted(
    zip(X.columns, coef),
    key=lambda x: abs(x[1]),
    reverse=True
)
print("Feature priority (high → low):")
for f, c in importance:
    print(f"{f}: {c:.4f}")

# Save model
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
joblib.dump(pipeline, MODEL_PATH)
print("Model saved at:", MODEL_PATH)
