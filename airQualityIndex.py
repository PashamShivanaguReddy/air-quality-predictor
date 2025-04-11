import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import folium
from streamlit_folium import folium_static
from tensorflow.keras.models import load_model
import joblib

# ============ API Key ============
API_KEY = "d14a4f432f95fbcc237c73076e774343"

# ============ Load Pre-trained Model ============
model = load_model("aqi_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# ============ LSTM Prediction Function ============

def predict_future(model, past_df, steps=4):
    data = scaler.transform(past_df[['pm2_5', 'pm10', 'so2', 'no2']])
    predictions = []

    input_seq = data[-3:].copy()
    for _ in range(steps):
        input_seq_reshaped = np.expand_dims(input_seq, axis=0)
        pred = model.predict(input_seq_reshaped, verbose=0)[0]
        predictions.append(pred)
        input_seq = np.vstack([input_seq[1:], pred])
    
    predictions = scaler.inverse_transform(predictions)
    dates = pd.date_range(datetime.now(), periods=steps).date
    return pd.DataFrame(predictions, columns=["pm2_5", "pm10", "so2", "no2"], index=dates)

# ============ API Functions ============

def get_coordinates(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    res = requests.get(url).json()
    return (res['coord']['lat'], res['coord']['lon']) if 'coord' in res else (None, None)

def get_current_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    res = requests.get(url).json()
    return {
        "temp": res['main']['temp'],
        "humidity": res['main']['humidity'],
        "wind_speed": res['wind']['speed'],
        "wind_deg": res['wind']['deg']
    }

def get_air_quality(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    res = requests.get(url).json()
    return pd.DataFrame([{
        "datetime": pd.to_datetime(i['dt'], unit='s'),
        "pm2_5": i['components']['pm2_5'],
        "pm10": i['components']['pm10'],
        "so2": i['components']['so2'],
        "no2": i['components']['no2']
    } for i in res.get('list', [])])

def deg_to_direction(deg):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[round(deg / 45) % 8]

# ============ Health Advice & Chatbot ============

def get_suggestions(condition, pm2_5):
    if pm2_5 <= 12:
        status = "Good"
    elif pm2_5 <= 35:
        status = "Moderate"
    elif pm2_5 <= 55:
        status = "Unhealthy for Sensitive Groups"
    elif pm2_5 <= 150:
        status = "Unhealthy"
    else:
        status = "Very Unhealthy"

    recs = {
        "Asthma": "Carry inhaler, avoid exertion, wear a mask.",
        "Heart Disease": "Avoid exercise, stay indoors, wear a mask.",
        "Children": "Keep indoors, avoid outdoor play.",
        "Elderly": "Stay hydrated and indoors, wear a mask.",
        "Healthy": "Wear a mask on poor AQI days."
    }
    return status, recs.get(condition, "Avoid pollution exposure.")

def chatbot_response(msg, condition, pm2_5):
    msg = msg.lower()
    base = "✅ Air is okay." if pm2_5 <= 35 else (
        "⚠️ Moderate air. Sensitive groups, be cautious." if pm2_5 <= 55 else "🚫 Unhealthy air. Avoid going out.")

    recs = {
        "Asthma": "Use inhaler, stay indoors, wear a mask.",
        "Heart Disease": "Avoid exertion outdoors, stay cool, wear a mask.",
        "Children": "Limit outdoor activity.",
        "Elderly": "Avoid pollution and stay hydrated.",
        "Healthy": "Use mask and avoid long outdoor exposure."
    }

    if any(k in msg for k in ["hi", "hello", "hey"]):
        return f"👋 Hello! I'm your air quality bot. You selected '{condition}'. Ask away!"

    if any(k in msg for k in ["safe", "okay", "go out"]):
        return f"{base} Advice for {condition}: {recs.get(condition)}"

    if any(k in msg for k in ["precaution", "do", "mask"]):
        return f"😷 For {condition}: {recs.get(condition)}"

    return "🤖 Ask if it's safe to go out, or what precautions to take!"

# ============ Streamlit App ============

st.set_page_config("🌤️ Air Quality & Weather Advisor", layout="wide")
st.title("🌍 Smart Air Quality & Weather Assistant")

city = st.text_input("Enter a city name:")
health_condition = st.selectbox("Select your health condition:", ["Healthy", "Asthma", "Heart Disease", "Children", "Elderly"])

if city:
    lat, lon = get_coordinates(city)
    if lat:
        st.subheader("🌡️ Current Weather & 🗺️ City Map")
        col1, col2 = st.columns(2)

        with col1:
            weather = get_current_weather(lat, lon)
            st.metric("Temperature (°C)", weather["temp"])
            st.metric("Humidity (%)", weather["humidity"])
            st.metric("Wind Speed (m/s)", weather["wind_speed"])
            st.metric("Wind Direction", deg_to_direction(weather["wind_deg"]))

        with col2:
            m = folium.Map(location=[lat, lon], zoom_start=11)
            folium.Marker([lat, lon], tooltip=city).add_to(m)
            folium_static(m)

        st.subheader("📊 AQI Forecast (PM2.5, PM10, SO₂, NO₂)")
        aqi_df = get_air_quality(lat, lon)
        if not aqi_df.empty:
            aqi_df = aqi_df.set_index("datetime").resample("D").mean().reset_index()
            past_df = aqi_df.tail(7).copy()

            st.markdown("### 🔮 LSTM-based AQI Forecast")
            future_df = predict_future(model, past_df)

            full_df = pd.concat([
                past_df.set_index("datetime")[["pm2_5", "pm10", "so2", "no2"]],
                future_df.rename_axis("datetime")
            ])
            fig = px.line(full_df, x=full_df.index, y=full_df.columns, title="Predicted AQI (μg/m³)")
            st.plotly_chart(fig, use_container_width=True)

            latest_pm2_5 = future_df.iloc[0]['pm2_5']
            status, message = get_suggestions(health_condition, latest_pm2_5)
            st.success(f"**Predicted Air Quality:** {status}\n\n**Advice for {health_condition}:** {message}")

        st.subheader("🤖 Chatbot Assistant")
        st.markdown("Ask me something like:\n- Is it safe to go outside today?\n- What precautions should I take?\n- Do I need a mask?")
        user_msg = st.text_input("Your question:")
        if user_msg:
            st.write(chatbot_response(user_msg, health_condition, latest_pm2_5))








