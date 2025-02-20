import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# API Key (Replace with your actual OpenWeather API key)
API_KEY = 'd14a4f432f95fbcc237c73076e774343'

# Function to get city coordinates
def get_city_coordinates(city_name):
    geocode_url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}'
    try:
        response = requests.get(geocode_url)
        response.raise_for_status()
        data = response.json()
        lat = data['coord']['lat']
        lon = data['coord']['lon']
        return lat, lon
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coordinates for {city_name}: {e}")
        return None, None

# Function to get weather data
def get_weather_data(lat, lon):
    weather_url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    try:
        response = requests.get(weather_url)
        response.raise_for_status()
        data = response.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind'].get('deg', None),
            'icon': data['weather'][0]['icon']
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# Function to get hourly air quality
def get_hourly_air_quality(lat, lon):
    url = f'http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        hourly_data = []
        for hour in data['list']:
            timestamp = hour['dt']
            air_quality = hour['components']
            datetime = pd.to_datetime(timestamp, unit='s')
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
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching air quality data: {e}")
        return None

# Function to fetch and combine weather and air quality data
def fetch_combined_data(city_name):
    lat, lon = get_city_coordinates(city_name)
    if lat is None or lon is None:
        return None, None

    air_quality_data = get_hourly_air_quality(lat, lon)
    weather_data = get_weather_data(lat, lon)

    if air_quality_data and weather_data:
        df_air_quality = pd.DataFrame(air_quality_data)
        for col, value in weather_data.items():
            df_air_quality[col] = value
        return df_air_quality, weather_data
    else:
        return None, None

# Function to display weather icon
def display_weather_icon(icon_code):
    if icon_code:
        icon_url = f"http://openweathermap.org/img/wn/{icon_code}@2x.png"
        st.image(icon_url, width=100)
    else:
        st.write("No icon available")

# Function to categorize AQI
def get_aqi_category(pm2_5):
    if pm2_5 is None:
        return "Unknown", "gray"
    if pm2_5 <= 12:
        return "Good", "green"
    elif pm2_5 <= 35:
        return "Moderate", "yellow"
    elif pm2_5 <= 55:
        return "Unhealthy for Sensitive Groups", "orange"
    elif pm2_5 <= 150:
        return "Unhealthy", "red"
    elif pm2_5 <= 250:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

# Set up Streamlit page
st.set_page_config(
    page_title="Weather & Air Quality | Real-time Data & Insights", 
    page_icon="🌤️", 
    layout="wide", 
    initial_sidebar_state="expanded",
)

# Main function
def main():
    st.markdown("<h1 style='text-align: center;'>🌤️ Weather & Air Quality</h1>", unsafe_allow_html=True)

    city_name = st.text_input("Enter City Name:")

    if city_name:
        with st.spinner(f"Fetching data for {city_name}..."):
            combined_data, weather_data = fetch_combined_data(city_name)

        if combined_data is not None and weather_data is not None:
            col1, col2 = st.columns(2)

            # LEFT COLUMN - Weather Data
            with col1:
                st.markdown(f"<h2 style='text-align: center;'>Weather in {city_name} ☁️</h2>", unsafe_allow_html=True)
                display_weather_icon(weather_data.get('icon'))
                st.markdown(f"<h3 style='text-align: center;'>{weather_data['weather'].capitalize()}</h3>", unsafe_allow_html=True)

                # Temperature Gauge
                fig_temp = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=weather_data['temperature'],
                    title={'text': "Temperature (°C)"},
                    gauge={
                        'axis': {'range': [-30, 50]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [-30, 0], 'color': "blue"},
                            {'range': [0, 20], 'color': "lightblue"},
                            {'range': [20, 35], 'color': "yellow"},
                            {'range': [35, 50], 'color': "orange"}
                        ]
                    }
                ))
                st.plotly_chart(fig_temp)

                # Weather Metrics
                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                col_w1.metric("💧 Humidity (%)", weather_data['humidity'])
                col_w2.metric("🌡 Pressure (hPa)", weather_data['pressure'])
                col_w3.metric("💨 Wind Speed (m/s)", weather_data['wind_speed'])
                col_w4.metric("🧭 Wind Direction", f"{weather_data['wind_deg']}°")

            # RIGHT COLUMN - Air Quality Data
            with col2:
                st.markdown(f"<h2 style='text-align: center;'>Air Quality in {city_name} 💨</h2>", unsafe_allow_html=True)

                fig_air_quality = px.line(combined_data, x='datetime', y=['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3'], title='Air Quality Over Time')
                st.plotly_chart(fig_air_quality)

                last_data = combined_data.iloc[-1]
                pm2_5 = last_data['pm2_5']
                aqi_category, aqi_color = get_aqi_category(pm2_5)
                st.markdown(f"<div style='text-align: center; color: {aqi_color};'><b>Air Quality: {aqi_category}</b></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
