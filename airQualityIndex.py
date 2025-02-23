import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set Page Configuration
st.set_page_config(page_title="Weather & Air Quality", page_icon="🌤️", layout="wide")

# Custom CSS for Better UI
st.markdown("""
    <style>
    .metric-container {
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
        background-color: #1e1e1e;
        color: white;
        box-shadow: 3px 3px 10px rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# API Key (Replace with your actual OpenWeather API key)
API_KEY = 'd14a4f432f95fbcc237c73076e774343'

# Function to get city coordinates
def get_city_coordinates(city_name):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['coord']['lat'], data['coord']['lon']
    except:
        st.error("Invalid city name or API issue!")
        return None, None

# Function to get weather data
def get_weather_data(lat, lon):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    response = requests.get(url).json()
    return {
        'temperature': response['main']['temp'],
        'humidity': response['main']['humidity'],
        'pressure': response['main']['pressure'],
        'weather': response['weather'][0]['description'],
        'wind_speed': response['wind']['speed'],
        'wind_deg': response['wind'].get('deg', None),
        'icon': response['weather'][0]['icon']
    }

# Function to get air quality data
def get_hourly_air_quality(lat, lon):
    url = f'http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}'
    response = requests.get(url).json()
    hourly_data = []
    for hour in response['list']:
        timestamp = hour['dt']
        datetime = pd.to_datetime(timestamp, unit='s', utc=True)
        air_quality = hour['components']
        hourly_data.append({
            'datetime': datetime,
            'pm2_5': air_quality.get('pm2_5', None),
            'pm10': air_quality.get('pm10', None),
            'co': air_quality.get('co', None),
            'no2': air_quality.get('no2', None),
            'so2': air_quality.get('so2', None),
            'o3': air_quality.get('o3', None),
        })
    return hourly_data

# Function to predict PM2.5
def predict_pm2_5(df):
    df = df.dropna()
    if len(df) < 10:
        return None, None, None
    
    X = df[['pm10', 'co', 'no2', 'so2', 'o3']]
    y = df['pm2_5']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    future_data = X.tail(5).copy()
    future_data.index = pd.date_range(start=df['datetime'].max(), periods=5, freq='H')
    future_predictions = model.predict(future_data)

    return future_data.index, future_predictions, mae

# UI Section
st.title("🌤️ Weather & Air Quality Dashboard")
st.subheader("Get real-time air quality, weather updates, and PM2.5 predictions 📊")

city_name = st.text_input("🏙️ Enter City Name:", "")
if city_name:
    with st.spinner(f"Fetching data for {city_name}..."):
        lat, lon = get_city_coordinates(city_name)
        if lat is not None and lon is not None:
            air_quality_data = get_hourly_air_quality(lat, lon)
            weather_data = get_weather_data(lat, lon)
            
            if air_quality_data and weather_data:
                df = pd.DataFrame(air_quality_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"🌍 Weather in {city_name}")
                    st.image(f"http://openweathermap.org/img/wn/{weather_data['icon']}@2x.png", width=80)
                    st.markdown(f"<div class='metric-container'>🌡️ Temperature: {weather_data['temperature']}°C</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-container'>💧 Humidity: {weather_data['humidity']}%</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-container'>🌬️ Wind Speed: {weather_data['wind_speed']} m/s</div>", unsafe_allow_html=True)
                
                with col2:
                    st.subheader(f"🛑 Air Quality in {city_name}")
                    fig_air_quality = px.line(df, x='datetime', y=['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3'], title='Air Quality Over Time')
                    st.plotly_chart(fig_air_quality, use_container_width=True)
                    
                future_times, future_pm2_5, mae = predict_pm2_5(df)
                if future_pm2_5 is not None:
                    st.markdown(f"<div class='metric-container'>📊 Mean Absolute Error (MAE): {round(mae, 2)}</div>", unsafe_allow_html=True)
                    future_fig = go.Figure()
                    future_fig.add_trace(go.Scatter(x=future_times, y=future_pm2_5, mode='lines+markers', name='Predicted PM2.5'))
                    future_fig.update_layout(title="📈 Predicted PM2.5 for Next Hours", xaxis_title="Time", yaxis_title="PM2.5 Level")
                    st.plotly_chart(future_fig, use_container_width=True)






