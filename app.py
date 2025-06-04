import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Configuration
st.set_page_config(layout="wide", page_title="Wind Forecast Dashboard")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0E1117; color: white;}
    .stTextInput input, .stSelectbox select {background-color: #1E1E1E; color: white;}
    h1, h2, h3, h4, h5, h6 {color: white !important;}
    .st-bb, .st-at {background-color: #1E1E1E;}
    .metric-card {border-radius: 10px; padding: 15px; background-color: #1E1E1E; color: white;}
</style>
""", unsafe_allow_html=True)

# API Functions
@st.cache_data(ttl=3600)
def get_coordinates(location):
    """Get coordinates with validation"""
    if not location or len(location.strip()) < 2:
        return None, None, "Please enter a valid location name", None
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
        headers = {"User-Agent": "WindEnergyApp/1.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return None, None, f"Location '{location}' not found", None
            
        first = data[0]
        lat = float(first.get('lat', 0))
        lon = float(first.get('lon', 0))
        display_name = first.get('display_name', '')
        
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon, None, display_name.split(',')[0]
        return None, None, "Invalid coordinates received", None
    except Exception as e:
        return None, None, f"API Error: {str(e)}", None

@st.cache_data(ttl=3600)
def get_weather_data(lat, lon, past_days=5):
    """Get weather data with validation"""
    try:
        # Get past data
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days={past_days}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not all(key in data.get('hourly', {}) for key in ['wind_speed_10m', 'wind_direction_10m']):
            return {"error": "Incomplete weather data received"}
            
        return data
    except Exception as e:
        return {"error": str(e)}

# Wind Speed Prediction Model
def prepare_features(df):
    """Create features for prediction model"""
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['day_of_year'] = df['Time'].dt.dayofyear
    df['month'] = df['Time'].dt.month
    df['wind_speed_lag1'] = df['Wind Speed (m/s)'].shift(1)
    df['wind_speed_lag2'] = df['Wind Speed (m/s)'].shift(2)
    df['wind_speed_lag3'] = df['Wind Speed (m/s)'].shift(3)
    return df.dropna()

def train_model(df):
    """Train wind speed prediction model"""
    df = prepare_features(df.copy())
    X = df[['hour_sin', 'hour_cos', 'day_of_week', 'day_of_year', 'month',
            'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)',
            'wind_speed_lag1', 'wind_speed_lag2', 'wind_speed_lag3']]
    y = df['Wind Speed (m/s)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    test_accuracy = model.score(X_test, y_test)
    return model, X.columns.tolist(), test_accuracy

def predict_future(model, features, last_data, hours_to_predict):
    """Predict future wind speeds"""
    pred_data = []
    last_time = last_data['Time']
    
    for i in range(1, hours_to_predict+1):
        pred_time = last_time + timedelta(hours=i)
        
        row = {
            'Time': pred_time,
            'hour': pred_time.hour,
            'Temperature (°C)': last_data['Temperature (°C)'],
            'Humidity (%)': last_data['Humidity (%)'],
            'Pressure (hPa)': last_data['Pressure (hPa)'],
            'wind_speed_lag1': last_data['Wind Speed (m/s)'],
            'wind_speed_lag2': last_data.get('wind_speed_lag1', last_data['Wind Speed (m/s)']),
            'wind_speed_lag3': last_data.get('wind_speed_lag2', last_data['Wind Speed (m/s)'])
        }
        pred_data.append(row)
    
    pred_df = pd.DataFrame(pred_data)
    pred_df = prepare_features(pred_df)
    
    # Make sure we have all required features
    missing_features = set(features) - set(pred_df.columns)
    for f in missing_features:
        pred_df[f] = 0
    
    predictions = model.predict(pred_df[features])
    pred_df['Predicted Wind Speed (m/s)'] = predictions
    
    return pred_df[['Time', 'Predicted Wind Speed (m/s)']]

# Main App
def main():
    st.title("🌬️ Wind Speed Forecasting Dashboard")
    
    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("📍 Location", "New York, US")
    with col2:
        hours_to_predict = st.slider("Hours to predict", 6, 48, 24)
    
    if st.button("Get Forecast"):
        with st.spinner("Fetching data and training model..."):
            # Get coordinates
            lat, lon, error, display_name = get_coordinates(location)
            if error:
                st.error(f"❌ {error}")
                return
            
            st.success(f"📍 Location: {display_name} (Lat: {lat:.4f}, Lon: {lon:.4f})")
            
            # Get weather data (past 5 days)
            data = get_weather_data(lat, lon, past_days=5)
            if 'error' in data:
                st.error(f"❌ Weather API Error: {data['error']}")
                return
            
            # Create DataFrame
            times = pd.to_datetime(data['hourly']['time'])
            df = pd.DataFrame({
                "Time": times,
                "Wind Speed (m/s)": data['hourly']['wind_speed_10m'],
                "Wind Direction": data['hourly']['wind_direction_10m'],
                "Temperature (°C)": data['hourly']['temperature_2m'],
                "Humidity (%)": data['hourly']['relative_humidity_2m'],
                "Pressure (hPa)": data['hourly']['surface_pressure']
            })
            
            # Add hour column
            df['hour'] = df['Time'].dt.hour
            
            # Train model
            model, features, test_accuracy = train_model(df)
            
            # Get last data point for prediction
            last_data = df.iloc[-1].to_dict()
            
            # Make predictions
            pred_df = predict_future(model, features, last_data, hours_to_predict)
            
            # Combine historical and predicted data
            historical_df = df[['Time', 'Wind Speed (m/s)']].copy()
            historical_df['Type'] = 'Historical'
            
            predicted_df = pred_df.rename(columns={'Predicted Wind Speed (m/s)': 'Wind Speed (m/s)'})
            predicted_df['Type'] = 'Predicted'
            
            combined_df = pd.concat([historical_df, predicted_df])
            
            # Find the transition point
            transition_time = last_data['Time']
            
            # Visualization
            st.subheader("Wind Speed Forecast")
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_df['Time'],
                y=historical_df['Wind Speed (m/s)'],
                name='Historical Data',
                line=dict(color='#1f77b4', width=2),
                mode='lines'
            ))
            
            # Predicted data
            fig.add_trace(go.Scatter(
                x=predicted_df['Time'],
                y=predicted_df['Wind Speed (m/s)'],
                name='Predicted',
                line=dict(color='#ff7f0e', width=2, dash='dot'),
                mode='lines'
            ))
            
            # Confidence band (simple example)
            fig.add_trace(go.Scatter(
                x=predicted_df['Time'],
                y=predicted_df['Wind Speed (m/s)'] * 1.1,
                line=dict(width=0),
                showlegend=False,
                mode='lines'
            ))
            
            fig.add_trace(go.Scatter(
                x=predicted_df['Time'],
                y=predicted_df['Wind Speed (m/s)'] * 0.9,
                fill='tonexty',
                fillcolor='rgba(255,127,14,0.2)',
                line=dict(width=0),
                name='Confidence Range',
                mode='lines'
            ))
            
            # Add vertical line at transition
            fig.add_vline(
                x=transition_time,
                line_dash="dash",
                line_color="red",
                annotation_text="Forecast Start",
                annotation_position="top left"
            )
            
            fig.update_layout(
                title=f"Wind Speed Forecast for {display_name}",
                xaxis_title="Time",
                yaxis_title="Wind Speed (m/s)",
                template="plotly_dark",
                hovermode="x unified",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.subheader("Forecast Metrics")
            col1, col2, col3 = st.columns(3)
            
            # Historical stats
            avg_wind = historical_df['Wind Speed (m/s)'].mean()
            col1.metric("Historical Avg Wind Speed", f"{avg_wind:.2f} m/s")
            
            # Prediction stats
            pred_avg = predicted_df['Wind Speed (m/s)'].mean()
            col2.metric("Predicted Avg Wind Speed", f"{pred_avg:.2f} m/s")
            
            # Model accuracy
            col3.metric("Model Accuracy (R²)", f"{test_accuracy:.2%}")
            
            # Show transition clearly
            st.info(f"""
            **Forecast Transition Point**: {transition_time.strftime('%Y-%m-%d %H:%M')}
            - **Historical Data**: Up to {transition_time.strftime('%Y-%m-%d %H:%M')}
            - **Predicted Data**: From {transition_time.strftime('%Y-%m-%d %H:%M')} onward
            """)
            
            # Raw data
            with st.expander("View Raw Data"):
                st.dataframe(combined_df)

if __name__ == "__main__":
    main()
