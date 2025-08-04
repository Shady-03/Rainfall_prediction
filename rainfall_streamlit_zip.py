import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import zipfile
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Rainfall Prediction", layout="centered")
st.title("🌧️ Rainfall Prediction Dashboard")

# 📁 Upload rainfall.zip or use default rainfall.csv.zip
st.subheader("📁 Upload Zipped Rainfall Data (.zip)")
uploaded_file = st.file_uploader("Upload a ZIP file containing rainfall.csv", type=["zip"])

if uploaded_file is not None:
    with zipfile.ZipFile(uploaded_file) as z:
        with z.open("rainfall.csv") as f:
            df = pd.read_csv(f, parse_dates=["date"])
    st.success("✅ Zipped CSV file loaded successfully.")
else:
    with zipfile.ZipFile("rainfall.csv.zip") as z:
        with z.open("rainfall.csv") as f:
            df = pd.read_csv(f, parse_dates=["date"])
    st.info("ℹ️ Using default rainfall.csv.zip")

# Convert daily to monthly totals
df["month"] = df["date"].dt.to_period("M")
monthly = df.groupby("month")["rainfall"].sum().reset_index()
monthly["rainfall"] = monthly["rainfall"].fillna(0)
data = monthly

# Preprocessing
values = data["rainfall"].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

def create_lags(series, n_lags=7):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i, 0])
        y.append(series[i, 0])
    return np.array(X), np.array(y)

X, y = create_lags(scaled)

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

# Inverse transform predictions
actual = scaler.inverse_transform(y.reshape(-1, 1))
predicted = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Metrics
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)

st.subheader("📈 Model Performance")
st.markdown(f"**Mean Squared Error:** {mse:.2f}")
st.markdown(f"**R² Score:** {r2:.2f}")

# Plot predictions
st.subheader("📊 Actual vs Predicted Rainfall (First 100 Months)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(actual[:100], label="Actual")
ax.plot(predicted[:100], label="Predicted")
ax.set_xlabel("Month Index")
ax.set_ylabel("Rainfall (mm)")
ax.set_title("Rainfall Forecast")
ax.legend()
st.pyplot(fig)

# Feature importances
st.subheader("🧠 Feature Importances (Lag Months)")
importances = model.feature_importances_
fig2, ax2 = plt.subplots()
ax2.bar(range(len(importances)), importances)
ax2.set_title("Feature Importances")
ax2.set_xlabel("Lag Month (0 = Most Recent)")
ax2.set_ylabel("Importance")
st.pyplot(fig2)

# Show data
with st.expander("📂 Show Raw Monthly Data"):
    st.dataframe(data.tail(20))

# Download predictions
output_df = pd.DataFrame({
    "Actual Rainfall (mm)": actual.flatten(),
    "Predicted Rainfall (mm)": predicted.flatten()
})

st.download_button(
    label="📥 Download Predictions as CSV",
    data=output_df.to_csv(index=False),
    file_name="rainfall_predictions.csv",
    mime="text/csv"
)

# 🌐 API Integration (Temporarily Disabled)
# st.subheader("🌐 Actual vs Predicted Rainfall from External API")
# try:
#     with st.spinner("Fetching data from Flask API..."):
#         response = requests.get("http://127.0.0.1:8080/api/data")
#         if response.status_code == 200:
#             ...
# except Exception as e:
#     st.error(f"Failed to fetch API data: {e}")
