import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Rainfall Prediction", layout="centered")
st.title("🌧️ Rainfall Prediction Dashboard")

# 📁 Upload rainfall.csv or use default
st.subheader("📁 Upload Daily Rainfall Data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.success("✅ File uploaded successfully.")
else:
    df = pd.read_csv("rainfall.csv", parse_dates=["date"])
    st.info("ℹ️ Using default rainfall.csv")

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

# 🌐 API Integration
st.subheader("🌐 Actual vs Predicted Rainfall from External API")

API_URL = "http://127.0.0.1:8080/api/data"  # Replace with public URL when deployed

try:
    with st.spinner("Fetching data from Flask API..."):
        response = requests.get(API_URL)
        if response.status_code == 200:
            api_data = response.json()
            actual_api = api_data["actual"]
            predicted_api = api_data["predicted"]
            indices = list(range(len(actual_api)))

            st.success("✅ Data loaded from API")

            # 2D Plot
            st.markdown("### 📊 2D Line Plot")
            fig_api, ax_api = plt.subplots(figsize=(12, 5))
            ax_api.plot(actual_api, label="Actual")
            ax_api.plot(predicted_api, label="Predicted")
            ax_api.set_title("Rainfall Prediction (From API)")
            ax_api.set_xlabel("Month Index")
            ax_api.set_ylabel("Rainfall (mm)")
            ax_api.legend()
            st.pyplot(fig_api)

            # Toggle for 3D plot style
            st.markdown("### 🧊 3D Plot Options")
            plot_style = st.radio("Choose 3D graph style:", ["Single Color", "Dual Color"])

            if plot_style == "Single Color":
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=indices,
                    y=actual_api,
                    z=predicted_api,
                    mode='lines+markers',
                    line=dict(color='blue'),
                    marker=dict(size=3),
                    name='Actual vs Predicted'
                )])
            else:
                trace_actual = go.Scatter3d(
                    x=indices,
                    y=actual_api,
                    z=[0]*len(actual_api),
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='blue'),
                    marker=dict(size=3)
                )
                trace_predicted = go.Scatter3d(
                    x=indices,
                    y=[0]*len(predicted_api),
                    z=predicted_api,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red'),
                    marker=dict(size=3)
                )
                fig_3d = go.Figure(data=[trace_actual, trace_predicted])

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Month Index',
                    yaxis_title='Actual Rainfall (mm)',
                    zaxis_title='Predicted Rainfall (mm)'
                ),
                title="3D Actual vs Predicted Rainfall (From API)",
                margin=dict(l=0, r=0, b=0, t=40)
            )

            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.error(f"API returned error: {response.status_code}")
except Exception as e:
    st.error(f"Failed to fetch API data: {e}")
