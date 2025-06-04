import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Configuration
st.set_page_config(layout="wide", page_title="Wind Forecast Dashboard")

# Custom CSS for clean look
st.markdown("""
<style>
    .main {background-color: white; color: #333;}
    .stTextInput input, .stSelectbox select {border: 1px solid #ddd;}
    .st-bb {background-color: white;}
    .st-at {background-color: white;}
    h1, h2, h3 {color: #333;}
    .info-box {background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# API Functions
@st.cache_data(ttl=3600)
def get_coordinates(location):
    """Get coordinates with validation"""
    if not location or len(location.strip()) < 2:
        return None, None, "Please enter a valid location name"
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
        headers = {"User-Agent": "WindEnergyApp/1.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return None, None, f"Location '{location}' not found", None
            
        # Get the first result with valid coordinates
        for item in data:
            try:
                lat = float(item.get('lat', 0))
                lon = float(item.get('lon', 0))
                display_name = item.get('display_name', '')
                
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon, None, display_name.split(',')[0]
            except (ValueError, TypeError):
                continue
                
        return None, None, f"Couldn't find valid coordinates for '{location}'", None
    except Exception as e:
        return None, None, f"API Error: {str(e)}", None

@st.cache_data(ttl=3600)
def get_weather_data(lat, lon):
    """Get weather data for past 5 days and next 48 hours"""
    try:
        # Get historical data (past 5 days)
        hist_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m&past_days=5"
        hist_response = requests.get(hist_url)
        hist_response.raise_for_status()
        hist_data = hist_response.json()
        
        # Get forecast data (next 48 hours)
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m&forecast_days=2"
        forecast_response = requests.get(forecast_url)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        # Combine data
        combined_data = {
            'hourly': {
                'time': hist_data['hourly']['time'] + forecast_data['hourly']['time'],
                'wind_speed_10m': hist_data['hourly']['wind_speed_10m'] + forecast_data['hourly']['wind_speed_10m'],
                'wind_direction_10m': hist_data['hourly']['wind_direction_10m'] + forecast_data['hourly']['wind_direction_10m']
            }
        }
        
        return combined_data
    except Exception as e:
        return {"error": str(e)}

# Wind Speed Prediction Model
def train_wind_speed_model(df):
    # Feature engineering
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['day_of_year'] = df['Time'].dt.dayofyear
    
    # Lag features
    df['wind_speed_lag1'] = df['Wind Speed (m/s)'].shift(1)
    df['wind_speed_lag2'] = df['Wind Speed (m/s)'].shift(2)
    df['wind_speed_lag3'] = df['Wind Speed (m/s)'].shift(3)
    
    df = df.dropna()
    
    # Prepare data
    X = df[['hour_sin', 'hour_cos', 'day_of_week', 'day_of_year',
            'wind_speed_lag1', 'wind_speed_lag2', 'wind_speed_lag3']]
    y = df['Wind Speed (m/s)']
    
    # Train/test split (most recent 20% for testing)
    test_size = int(len(df) * 0.2)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate test accuracy
    test_accuracy = 1 - mean_absolute_error(y_test, model.predict(X_test)) / y_test.mean()
    
    return model, X.columns.tolist(), test_accuracy

def predict_future_wind(model, features, last_data_point, future_hours):
    # Create future timestamps
    future_times = [last_data_point['Time'] + timedelta(hours=i) for i in range(1, future_hours+1)]
    
    # Prepare prediction data
    pred_data = []
    for i, time in enumerate(future_times):
        # Use the last known wind speed values for lag features
        lag1 = last_data_point['Wind Speed (m/s)']
        lag2 = last_data_point['wind_speed_lag1'] if 'wind_speed_lag1' in last_data_point else lag1
        lag3 = last_data_point['wind_speed_lag2'] if 'wind_speed_lag2' in last_data_point else lag2
        
        row = {
            'hour_sin': np.sin(2 * np.pi * time.hour/24),
            'hour_cos': np.cos(2 * np.pi * time.hour/24),
            'day_of_week': time.weekday(),
            'day_of_year': time.timetuple().tm_yday,
            'wind_speed_lag1': lag1,
            'wind_speed_lag2': lag2,
            'wind_speed_lag3': lag3
        }
        pred_data.append(row)
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(pred_data)[features]
    
    # Make predictions
    predictions = model.predict(pred_df)
    
    return future_times, predictions

# Main App
def main():
    st.title("🌬️ Wind Speed Forecasting Dashboard")
    
    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("📍 Location", "New York, US")
    with col2:
        future_hours = st.slider("Hours to forecast", 6, 48, 24, step=6)
    
    if st.button("Generate Forecast"):
        with st.spinner("Fetching data and generating forecast..."):
            # Get location coordinates
            lat, lon, error, display_name = get_coordinates(location)
            
            if error:
                st.error(f"❌ {error}")
                return
                
            # Get weather data
            data = get_weather_data(lat, lon)
            if 'error' in data:
                st.error(f"❌ Weather API Error: {data['error']}")
                return
            
            # Process data
            times = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in data['hourly']['time']]
            df = pd.DataFrame({
                "Time": times,
                "Wind Speed (m/s)": data['hourly']['wind_speed_10m'],
                "Wind Direction": data['hourly']['wind_direction_10m']
            })
            
            # Split into historical and forecast data
            now = datetime.now()
            historical_df = df[df['Time'] < now].copy()
            forecast_df = df[df['Time'] >= now].copy()
            
            # Train model on historical data
            model, features, test_accuracy = train_wind_speed_model(historical_df.copy())
            
            # Make predictions for future
            last_data_point = historical_df.iloc[-1].to_dict()
            future_times, future_wind = predict_future_wind(model, features, last_data_point, future_hours)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Time': future_times,
                'Predicted Wind Speed (m/s)': future_wind,
                'Type': 'Prediction'
            })
            
            # Combine all data for visualization
            historical_df['Type'] = 'Historical'
            forecast_df['Type'] = 'API Forecast'
            
            # Limit forecast data to 48 hours
            forecast_df = forecast_df[forecast_df['Time'] <= now + timedelta(hours=48)]
            
            # Combine all data
            combined_df = pd.concat([
                historical_df[['Time', 'Wind Speed (m/s)', 'Type']],
                forecast_df[['Time', 'Wind Speed (m/s)', 'Type']],
                pred_df.rename(columns={'Predicted Wind Speed (m/s)': 'Wind Speed (m/s)'})
            ])
            
            # Visualization
            st.subheader(f"Wind Speed Forecast for {display_name}")
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_df['Time'],
                y=historical_df['Wind Speed (m/s)'],
                name='Historical Data',
                line=dict(color='#636EFA'),
                mode='lines'
            ))
            
            # API Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Time'],
                y=forecast_df['Wind Speed (m/s)'],
                name='API Forecast',
                line=dict(color='#00CC96'),
                mode='lines'
            ))
            
            # Model Predictions
            fig.add_trace(go.Scatter(
                x=pred_df['Time'],
                y=pred_df['Predicted Wind Speed (m/s)'],
                name='Model Prediction',
                line=dict(color='#EF553B', dash='dot'),
                mode='lines'
            ))
            
            # Add vertical line for current time
            fig.add_vline(
                x=now.timestamp() * 1000,
                line_dash="dash",
                line_color="gray",
                annotation_text="Now",
                annotation_position="top left"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Date/Time",
                yaxis_title="Wind Speed (m/s)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            st.subheader("Forecast Summary")
            col1, col2, col3 = st.columns(3)
            
            # Current wind speed
            current_speed = historical_df.iloc[-1]['Wind Speed (m/s)']
            col1.metric("Current Wind Speed", f"{current_speed:.1f} m/s")
            
            # Average predicted speed
            avg_pred = pred_df['Predicted Wind Speed (m/s)'].mean()
            col2.metric("Average Predicted Speed", f"{avg_pred:.1f} m/s")
            
            # Model accuracy
            col3.metric("Model Accuracy", f"{test_accuracy:.1%}")
            
            # Data explanation
            with st.expander("About this forecast"):
                st.markdown("""
                - **Historical Data**: Actual wind measurements from the past 5 days
                - **API Forecast**: Official weather forecast for the next 48 hours
                - **Model Prediction**: Machine learning predictions beyond the API forecast range
                
                The prediction model uses past wind patterns to forecast future speeds, 
                achieving an accuracy of {:.1%} on test data.
                """.format(test_accuracy))

if __name__ == "__main__":
    main()
