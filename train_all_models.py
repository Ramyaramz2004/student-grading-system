# train_all_models.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# === Define dataset paths ===
STUDENT_DATA = os.path.join("..", "dataset", "students_dataset.csv")
MENTAL_DATA = os.path.join("..", "dataset", "mental_health_datasets.csv")

# === Create model directory ===
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Train Grade Prediction Model ===
def train_grade_model():
    if not os.path.exists(STUDENT_DATA):
        print(f"❌ Missing dataset: {STUDENT_DATA}")
        return

    print(f"📘 Loading academic dataset: {os.path.abspath(STUDENT_DATA)}")
    df = pd.read_csv(STUDENT_DATA)

    X = df[['assignment_marks', 'quiz_marks', 'attendance']]
    y = df['final_grade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, "student_grade_model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Grade model saved to {model_path}\n")

# === Train Mental Health Model ===
def train_mental_health_model():
    if not os.path.exists(MENTAL_DATA):
        print(f"❌ Missing dataset: {MENTAL_DATA}")
        return

    print(f"📗 Loading mental health dataset: {os.path.abspath(MENTAL_DATA)}")
    df = pd.read_csv(MENTAL_DATA)

    X = df[['study_hours', 'sleep_hours', 'stress_level', 'attendance']]
    y = df['mental_health_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, "mental_health_model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Mental health model saved to {model_path}\n")

# === Run both ===
if __name__ == "__main__":
    print("🚀 Starting training for all models...\n")
    train_grade_model()
    train_mental_health_model()
    print("🎉 All models trained successfully!")
