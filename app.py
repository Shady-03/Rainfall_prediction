# app.py (full version with static + frontend serving)

from flask import Flask, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("rainfall_model.pkl")
scaler = joblib.load("rainfall_scaler.pkl")

# === Rainfall data preprocessing ===
df = pd.read_csv("rainfall.csv", parse_dates=['date'])
df['month'] = df['date'].dt.to_period('M')
monthly_rainfall = df.groupby('month')['rainfall'].sum().reset_index()
monthly_rainfall['rainfall'] = monthly_rainfall['rainfall'].fillna(0)

# Normalize
rainfall_values = monthly_rainfall['rainfall'].values.reshape(-1, 1)
rainfall_scaled = scaler.transform(rainfall_values)

# Create lag features
def create_features(data, n_lags=7):
    X = []
    for i in range(n_lags, len(data)):
        X.append(data[i - n_lags:i, 0])
    return np.array(X)

X_input = create_features(rainfall_scaled)
pred_scaled = model.predict(X_input)
pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
actual = rainfall_values[7:]

# Save predictions to static/rainfall_data.json
data_to_save = {
    'actual': actual.flatten().tolist(),
    'predicted': pred_actual.flatten().tolist()
}

os.makedirs("static", exist_ok=True)
with open("static/rainfall_data.json", "w") as f:
    json.dump(data_to_save, f)

# === ROUTES ===

# Serve API JSON
@app.route("/api/data")
def get_data():
    return jsonify(data_to_save)

# Serve index.html (dashboard)
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

# Serve CSS, JS, and other static frontend files
@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("frontend", path)

# Start Flask server
if __name__ == "__main__":
    app.run(port=8080, debug=True)
