# ===== Importing Required Libraries =====
import streamlit as st                            # For UI elements
import requests                                   # For making API calls
import pandas as pd                               # For handling dataframes
import numpy as np                                # For numerical operations
from datetime import datetime                     # For handling dates
import plotly.express as px                       # For plotting interactive charts
import folium                                     # For rendering city map
from streamlit_folium import folium_static        # To embed folium maps in Streamlit
from sklearn.preprocessing import MinMaxScaler    # For scaling data
from tensorflow.keras.models import Sequential    # For building LSTM model
from tensorflow.keras.layers import LSTM, Dense   # LSTM and Dense layers

# ===== API Key for OpenWeatherMap =====
API_KEY = "d14a4f432f95fbcc237c73076e774343"

# ======= LSTM Helper Functions =======

# Prepares data by scaling and forming input-output sequences
def prepare_data(df, steps=3):
    df_scaled = scaler.fit_transform(df)          # Scale input features
    X, y = [], []
    for i in range(len(df_scaled) - steps):       # Create sliding window sequences
        X.append(df_scaled[i:i+steps])            # Last 'steps' rows as features
        y.append(df_scaled[i+steps])              # Next row as label
    return np.array(X), np.array(y)

# Trains LSTM model on past AQI data (7 days)
@st.cache_resource   
def train_lstm_model(past_df):
    global scaler
    scaler = MinMaxScaler()
    X, y = prepare_data(past_df[['pm2_5', 'pm10', 'so2', 'no2']])

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(4))                           # 4 outputs: pm2.5, pm10, so2, no2
    model.compile(optimizer='adam', loss='mse')   # Compile model
    model.fit(X, y, epochs=20, verbose=0)         # Train model silently
    return model, scaler

# Predicts AQI for the next 4 days using the LSTM model
def predict_future(model, past_df, steps=4):
    data = scaler.transform(past_df[['pm2_5', 'pm10', 'so2', 'no2']])
    predictions = []

    input_seq = data[-3:].copy()                  # Start with last 3 days of data
    for _ in range(steps):                        # Predict 4 future days
        input_seq_reshaped = np.expand_dims(input_seq, axis=0)
        pred = model.predict(input_seq_reshaped, verbose=0)[0]
        predictions.append(pred)
        input_seq = np.vstack([input_seq[1:], pred])  # Slide window forward
    
    predictions = scaler.inverse_transform(predictions)   # Convert back to original scale
    dates = pd.date_range(datetime.now(), periods=steps).date
    return pd.DataFrame(predictions, columns=["pm2_5", "pm10", "so2", "no2"], index=dates)

# ======= API Functions =======

# Gets the latitude and longitude of a city
def get_coordinates(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    res = requests.get(url).json()
    return (res['coord']['lat'], res['coord']['lon']) if 'coord' in res else (None, None)

# Gets current weather data for a given location
def get_current_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    res = requests.get(url).json()
    return {
        "temp": res['main']['temp'],
        "humidity": res['main']['humidity'],
        "wind_speed": res['wind']['speed'],
        "wind_deg": res['wind']['deg']
    }

# Gets air quality forecast (hourly) and converts it into a DataFrame
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

# Converts wind direction in degrees to human-readable form
def deg_to_direction(deg):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[round(deg / 45) % 8]

# ======= Health Advice Functions =======

# Maps pm2.5 to air quality category and gives advice
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

    # Health-specific advice
    recs = {
        "Asthma": "Carry inhaler, avoid exertion, wear a mask.",
        "Heart Disease": "Avoid exercise, stay indoors, wear a mask.",
        "Children": "Keep indoors, avoid outdoor play.",
        "Elderly": "Stay hydrated and indoors, wear a mask.",
        "Healthy": "Wear a mask on poor AQI days."
    }
    return status, recs.get(condition, "Avoid pollution exposure.")

# AI-like chatbot to answer user queries based on AQI and condition
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

    # Greeting intent
    if any(k in msg for k in ["hi", "hello", "hey"]):
        return f"👋 Hello! I'm your air quality bot. You selected '{condition}'. Ask away!"

    # Ask about safety outside
    if any(k in msg for k in ["safe", "okay", "go out"]):
        return f"{base} Advice for {condition}: {recs.get(condition)}"

    # Ask for precautions
    if any(k in msg for k in ["precaution", "do", "mask"]):
        return f"😷 For {condition}: {recs.get(condition)}"

    return "🤖 Ask if it's safe to go out, or what precautions to take!"

# ======= Streamlit App UI =======

# Page setup
st.set_page_config("🌤️ Air Quality & Weather Advisor", layout="wide")
st.title("🌍 Smart Air Quality & Weather Assistant")

# Input city and health condition
city = st.text_input("Enter a city name:")
health_condition = st.selectbox("Select your health condition:", ["Healthy", "Asthma", "Heart Disease", "Children", "Elderly"])

if city:
    lat, lon = get_coordinates(city)
    if lat:
        st.subheader("🌡️ Current Weather & 🗺️ City Map")
        col1, col2 = st.columns(2)

        # Show weather metrics
        with col1:
            weather = get_current_weather(lat, lon)
            st.metric("Temperature (°C)", weather["temp"])
            st.metric("Humidity (%)", weather["humidity"])
            st.metric("Wind Speed (m/s)", weather["wind_speed"])
            st.metric("Wind Direction", deg_to_direction(weather["wind_deg"]))

        # Show city map
        with col2:
            m = folium.Map(location=[lat, lon], zoom_start=11)
            folium.Marker([lat, lon], tooltip=city).add_to(m)
            folium_static(m)

        # Fetch and forecast AQI
        st.subheader("📊 AQI Forecast (PM2.5, PM10, SO₂, NO₂)")
        aqi_df = get_air_quality(lat, lon)
        if not aqi_df.empty:
            # Average to daily AQI
            aqi_df = aqi_df.set_index("datetime").resample("D").mean().reset_index()
            past_df = aqi_df.tail(7).copy()

            # LSTM Forecast
            st.markdown("### 🔮 LSTM-based AQI Forecast")
            model, scaler = train_lstm_model(past_df)
            future_df = predict_future(model, past_df)

            # Plot past + future
            full_df = pd.concat([past_df.set_index("datetime")[["pm2_5", "pm10", "so2", "no2"]],
                                 future_df.rename_axis("datetime")])
            fig = px.line(full_df, x=full_df.index, y=full_df.columns, title="Predicted AQI (μg/m³)")
            st.plotly_chart(fig, use_container_width=True)

            # Health-based suggestion
            latest_pm2_5 = future_df.iloc[0]['pm2_5']
            status, message = get_suggestions(health_condition, latest_pm2_5)
            st.success(f"**Predicted Air Quality:** {status}\n\n**Advice for {health_condition}:** {message}")

        # ===== Chatbot UI =====
        st.subheader("🤖 Chatbot Assistant")
        st.markdown("Ask me something like:\n- Is it safe to go outside today?\n- What precautions should I take?\n- Do I need a mask?")
        user_msg = st.text_input("Your question:")
        if user_msg:
            st.write(chatbot_response(user_msg, health_condition, latest_pm2_5))
