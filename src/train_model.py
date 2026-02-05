import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
DATA_PATH = os.path.join(BASE_DIR, "data", "students.csv")
df = pd.read_csv(DATA_PATH)

X = df.drop("placed", axis=1)
y = df["placed"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
joblib.dump(model, MODEL_PATH)
print("Model saved at:", MODEL_PATH)
