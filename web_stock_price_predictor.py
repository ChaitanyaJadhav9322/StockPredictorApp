import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model  
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Streamlit Page Setup
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

st.title("📊 Stock Price Predictor App")

# User Input
stock = st.text_input("Enter Stock Symbol (e.g., GOOG, AAPL, TSLA):", "GOOG")

# Load Data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start, end)

if google_data.empty:
    st.error("⚠️ Could not fetch data. Please check the stock symbol or your internet connection.")
    st.stop()

st.subheader("📄 Stock Data")
st.dataframe(google_data.tail())

# Load Model
try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"❌ Model could not be loaded: {e}")
    st.stop()

# Prepare Data
data = google_data[['Close']].values
train_len = int(len(data) * 0.7)
train_data = data[:train_len]
test_data = data[train_len - 100:]  # include overlap for prediction continuity

# Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

# Create Test Sequences
x_test, y_test = [], []
for i in range(100, len(scaled_test)):
    x_test.append(scaled_test[i - 100:i])
    y_test.append(scaled_test[i])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
predictions = model.predict(x_test)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

# Create DataFrame for Visualization
plot_data = pd.DataFrame(
    {
        'Actual Price': inv_y_test.reshape(-1),
        'Predicted Price': inv_predictions.reshape(-1)
    },
    index=google_data.index[train_len:]
)

# Show DataFrame
st.subheader("📉 Predicted vs Actual Stock Prices")
st.dataframe(plot_data.tail())

# Chart 1: Stock Close Price
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=google_data.index, y=google_data['Close'],
    mode='lines', name='Close Price',
    line=dict(color='blue', width=2)
))
fig1.update_layout(
    title=f"{stock} Stock Close Price Over Time",
    xaxis_title="Date", yaxis_title="Price (USD)",
    template="plotly_white"
)
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Prediction vs Actual
fig2 = make_subplots()
fig2.add_trace(go.Scatter(
    x=plot_data.index, y=plot_data['Actual Price'],
    mode='lines', name='Actual Price', line=dict(color='green', width=2)
))
fig2.add_trace(go.Scatter(
    x=plot_data.index, y=plot_data['Predicted Price'],
    mode='lines', name='Predicted Price', line=dict(color='red', width=2)
))
fig2.update_layout(
    title=f"{stock} - Predicted vs Actual Prices",
    xaxis_title="Date", yaxis_title="Price (USD)",
    template="plotly_white"
)
st.plotly_chart(fig2, use_container_width=True)
