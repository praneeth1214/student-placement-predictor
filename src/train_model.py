import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "students.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
X = df.drop("placed", axis=1)
y = df["placed"]

# --------------------------------------------------
# Train / Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Logistic Regression Pipeline (Base Model)
# --------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l2",
        C=0.5,
        solver="lbfgs",
        max_iter=1000
    ))
])

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# Evaluation (Logistic Regression)
# --------------------------------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(
    pipeline, X, y, cv=5, scoring="roc_auc"
)
print("\nCross-Validation ROC-AUC:", cv_scores)
print("Mean CV ROC-AUC:", cv_scores.mean())

# --------------------------------------------------
# Random Forest (Comparison only)
# --------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    random_state=42
)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
print("\nRandom Forest ROC-AUC:", roc_auc_score(y_test, rf_prob))

# --------------------------------------------------
# Probability Calibration (FINAL MODEL)
# --------------------------------------------------
calibrated_model = CalibratedClassifierCV(
    base_estimator=pipeline,
    method="sigmoid",
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Feature influence (for verification)
coef = calibrated_model.base_estimator.named_steps["model"].coef_[0]
importance = sorted(
    zip(X.columns, coef),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("\nFeature influence (high → low):")
for f, c in importance:
    print(f"{f}: {c:.4f}")

# --------------------------------------------------
# Save final model
# --------------------------------------------------
joblib.dump(calibrated_model, MODEL_PATH)
print("\nFinal calibrated model saved at:", MODEL_PATH)
