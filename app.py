from flask import Flask, request, render_template, send_file
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# === Load both trained models ===
grade_model = joblib.load(os.path.join("model", "student_grade_model.joblib"))
mental_model = joblib.load(os.path.join("model", "mental_health_model.joblib"))

# === Ensure folder for saving predictions ===
os.makedirs("predictions", exist_ok=True)
PRED_FILE = os.path.join("predictions", "predicted_results.xlsx")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user inputs
        name = request.form['name']
        age = int(request.form['age'])
        assignment = float(request.form['assignment'])
        quiz = float(request.form['quiz'])
        attendance = float(request.form['attendance'])
        study_hours = float(request.form['study_hours'])
        sleep_hours = float(request.form['sleep_hours'])
        stress_level = float(request.form['stress_level'])

        # === Predict Final Grade ===
        grade_pred = grade_model.predict([[assignment, quiz, attendance]])[0]
        grade_percent = min(max(grade_pred, 0), 100)

        # === Predict Mental Health ===
        mental_pred = mental_model.predict([[study_hours, sleep_hours, stress_level, attendance]])[0]

        # === Generate Recommendation ===
        if mental_pred == "Balanced":
            recommendation = "Great job maintaining a healthy balance!"
        elif mental_pred == "Mild Stress":
            recommendation = "Try short breaks and maintain consistent sleep habits."
        else:
            recommendation = "You seem stressed — try relaxation, and reach out for help if needed."

        # === Save prediction into Excel ===
        result = {
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Name': [name],
            'Age': [age],
            'Assignment Marks': [assignment],
            'Quiz Marks': [quiz],
            'Attendance (%)': [attendance],
            'Predicted Grade': [round(grade_pred, 2)],
            'Study Hours': [study_hours],
            'Sleep Hours': [sleep_hours],
            'Stress Level': [stress_level],
            'Mental Health': [mental_pred],
            'Recommendation': [recommendation]
        }

        df_new = pd.DataFrame(result)

        # Append to file if it exists
        if os.path.exists(PRED_FILE):
            old = pd.read_excel(PRED_FILE)
            combined = pd.concat([old, df_new], ignore_index=True)
        else:
            combined = df_new

        combined.to_excel(PRED_FILE, index=False)

        return render_template(
            'index.html',
            grade_text=f"{name} (Age {age}) — Predicted Grade: {grade_pred:.2f}",
            mental_text=f"Mental Health: {mental_pred} | Recommendation: {recommendation}",
            grade_percent=grade_percent
        )

    except Exception as e:
        return render_template('index.html', grade_text=f"Error: {str(e)}")

@app.route('/download')
def download_predictions():
    """Download all predictions as Excel file"""
    if os.path.exists(PRED_FILE):
        return send_file(PRED_FILE, as_attachment=True)
    else:
        return "No predictions found yet!"

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)