import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta

# Set Streamlit page config
st.set_page_config(page_title="📈 Stock Price Prediction using LSTM", layout="centered")

# Title
st.title("📈 Stock Price Prediction using LSTM")
st.caption("Predict stock prices using a trained LSTM model")

# Input form
with st.form("input_form"):
    stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, GOOGL, INFY):", "AAPL")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.today())

    submitted = st.form_submit_button("Predict")

if submitted:
    # Load the data
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            st.error("⚠️ No data found for the given stock symbol and date range.")
        else:
            st.success("✅ Data loaded!")
            
            # Preprocess
            data = df[['Close']].copy()
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Create sequences (lookback of 60 days)
            X_test, y_test = [], []
            for i in range(60, len(scaled_data)):
                X_test.append(scaled_data[i-60:i, 0])
                y_test.append(scaled_data[i, 0])
            X_test, y_test = np.array(X_test), np.array(y_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Load the trained model
            model_path = "/content/drive/MyDrive/DL_Projects/3.Stock Price Prediction LSTM/stock_price_lstm_model.h5"
            model = load_model(model_path)

            # Predict
            predicted_prices = model.predict(X_test)
            predicted_prices = scaler.inverse_transform(predicted_prices)
            real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Dates for prediction
            prediction_dates = df.index[60:]

            # Create DataFrame
            pred_df = pd.DataFrame({
                "Date": prediction_dates,
                "Real Price": real_prices.flatten(),
                "Predicted Price": predicted_prices.flatten()
            })

            # Plot
            st.subheader("📊 Predicted vs Actual Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(pred_df["Date"], pred_df["Real Price"], label="Real Price", color="blue")
            ax.plot(pred_df["Date"], pred_df["Predicted Price"], label="Predicted Price", color="red")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Display table
            st.subheader("📋 Prediction Table (last 10 entries)")
            st.dataframe(pred_df.tail(10).reset_index(drop=True))

    except Exception as e:
        st.error(f"❌ Error: {e}")
