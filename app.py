import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Configuration
st.set_page_config(layout="wide", page_title="Wind Energy Analytics Dashboard")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0E1117; color: white;}
    h1, h2, h3 {color: white !important;}
    .st-bb {background-color: #1E1E1E;}
    .st-at {background-color: #1E1E1E;}
    .metric-card {border-radius: 10px; padding: 15px; background-color: #1E1E1E; margin: 10px;}
</style>
""", unsafe_allow_html=True)

# API Functions
@st.cache_data(ttl=3600)
def get_coordinates(location):
    """Get coordinates for a location"""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
        headers = {"User-Agent": "WindForecastApp/1.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None, None, "Location not found", None
        first = data[0]
        return float(first['lat']), float(first['lon']), None, first['display_name']
    except Exception as e:
        return None, None, str(e), None

@st.cache_data(ttl=3600)
def get_weather_data(lat, lon, days=5):
    """Get historical and forecast weather data"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days={days-1}&forecast_days=2"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Wind Speed Prediction Model
def prepare_features(df):
    """Prepare features for prediction model"""
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['day_of_year'] = df['Time'].dt.dayofyear
    df['month'] = df['Time'].dt.month
    df['wind_speed_lag1'] = df['wind_speed_10m'].shift(1)
    df['wind_speed_lag2'] = df['wind_speed_10m'].shift(2)
    df['wind_speed_lag3'] = df['wind_speed_10m'].shift(3)
    return df.dropna()

def train_model(df):
    """Train wind speed prediction model"""
    features = ['hour_sin', 'hour_cos', 'day_of_week', 'day_of_year', 'month',
                'temperature_2m', 'relative_humidity_2m', 'surface_pressure',
                'wind_speed_lag1', 'wind_speed_lag2', 'wind_speed_lag3']
    
    X = df[features]
    y = df['wind_speed_10m']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, features, mae, r2

def main():
    st.title("🌬️ Wind Energy Analytics Dashboard")
    
    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("📍 Enter Location", "New York, US")
    with col2:
        turbine_model = st.selectbox("🌀 Turbine Model", 
                                  ["Vestas V80-2.0MW", "GE 1.5sle", "Suzlon S88-2.1MW"])
    
    if st.button("Get Forecast"):
        with st.spinner("Fetching data and training model..."):
            # Get coordinates
            lat, lon, error, display_name = get_coordinates(location)
            if error:
                st.error(f"Error: {error}")
                return
            
            # Get weather data (5 days total: 4 past + 1 forecast)
            data = get_weather_data(lat, lon, days=5)
            if 'error' in data:
                st.error(f"Weather API Error: {data['error']}")
                return
            
            # Prepare DataFrame
            df = pd.DataFrame({
                "Time": pd.to_datetime(data['hourly']['time']),
                "wind_speed_10m": data['hourly']['wind_speed_10m'],
                "wind_direction_10m": data['hourly']['wind_direction_10m'],
                "temperature_2m": data['hourly']['temperature_2m'],
                "relative_humidity_2m": data['hourly']['relative_humidity_2m'],
                "surface_pressure": data['hourly']['surface_pressure']
            })
            
            # Add air density calculation
            df['air_density'] = (df['surface_pressure'] * 100) / (287.05 * (df['temperature_2m'] + 273.15)) * \
                              (1 - (0.378 * (df['relative_humidity_2m']/100) * 0.61121 * 
                               np.exp((18.678 - df['temperature_2m']/234.5) * 
                               (df['temperature_2m']/(257.14 + df['temperature_2m'])))) / 
                               (df['surface_pressure'] * 100))
            
            # Split into past and future
            now = datetime.utcnow()
            past_df = df[df['Time'] <= now]
            future_df = df[df['Time'] > now]
            
            # Prepare features and train model
            model_df = prepare_features(past_df.copy())
            model, features, mae, r2 = train_model(model_df)
            
            # Make predictions for future
            if len(future_df) > 0:
                future_df = prepare_features(future_df.copy())
                future_df['predicted_wind'] = model.predict(future_df[features])
                future_df['predicted_power'] = future_df['predicted_wind'].apply(
                    lambda x: min(2000, max(0, 2000 * ((x - 4)/(15 - 4))**3) if 4 <= x <= 15 else (2000 if 15 < x <= 25 else 0)
                )
            
            # Combine past and future
            combined_df = pd.concat([
                past_df.assign(Type="Historical"),
                future_df.assign(Type="Forecast", 
                               wind_speed_10m=future_df['predicted_wind'],
                               power_output=future_df['predicted_power'])
            ])
            
            # Filter to last 7 days + next 48 hours
            start_date = now - timedelta(days=7)
            end_date = now + timedelta(hours=48)
            filtered_df = combined_df[(combined_df['Time'] >= start_date) & (combined_df['Time'] <= end_date)]
            
            # Calculate metrics
            avg_wind = filtered_df[filtered_df['Type'] == "Forecast"]['wind_speed_10m'].mean()
            max_wind = filtered_df[filtered_df['Type'] == "Forecast"]['wind_speed_10m'].max()
            total_energy = filtered_df[filtered_df['Type'] == "Forecast"]['power_output'].sum()
            
            # Display metrics
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average Wind Speed", f"{avg_wind:.2f} m/s")
            col2.metric("Peak Wind Speed", f"{max_wind:.2f} m/s")
            col3.metric("Total Energy", f"{total_energy/1000:.2f} MWh")
            col4.metric("Model Accuracy", f"{r2:.2%}")
            
            # Main forecast chart
            st.subheader("Wind Speed Forecast")
            fig = go.Figure()
            
            # Historical data
            hist_df = filtered_df[filtered_df['Type'] == "Historical"]
            fig.add_trace(go.Scatter(
                x=hist_df['Time'],
                y=hist_df['wind_speed_10m'],
                name='Historical Data',
                line=dict(color='#1f77b4', width=2),
                mode='lines'
            ))
            
            # Forecast data
            if len(future_df) > 0:
                forecast_df = filtered_df[filtered_df['Type'] == "Forecast"]
                fig.add_trace(go.Scatter(
                    x=forecast_df['Time'],
                    y=forecast_df['wind_speed_10m'],
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=3, dash='dot')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df['Time'],
                    y=forecast_df['wind_speed_10m'] * 1.1,
                    line=dict(width=0),
                    showlegend=False,
                    mode='lines'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['Time'],
                    y=forecast_df['wind_speed_10m'] * 0.9,
                    fill='tonexty',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(width=0),
                    name='Confidence Interval',
                    mode='lines'
                ))
            
            # Current time marker
            fig.add_vline(x=now.timestamp() * 1000, line_dash="dash", line_color="white", annotation_text="Now")
            
            fig.update_layout(
                title=f"Wind Speed Forecast for {display_name}",
                xaxis_title="Time",
                yaxis_title="Wind Speed (m/s)",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional charts in tabs
            tab1, tab2, tab3 = st.tabs(["Wind Direction", "Power Output", "Air Density"])
            
            with tab1:
                st.subheader("Wind Direction Analysis")
                fig = px.scatter_polar(filtered_df, r="wind_speed_10m", theta="wind_direction_10m",
                                      color="Type", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Power Output Forecast")
                fig = go.Figure()
                if len(future_df) > 0:
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Time'],
                        y=forecast_df['power_output'],
                        name='Power Output',
                        line=dict(color='#2ca02c', width=3)
                    ))
                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Time",
                    yaxis_title="Power Output (kW)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Air Density Analysis")
                fig = px.scatter(filtered_df, x="Time", y="air_density", 
                               color="temperature_2m", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
