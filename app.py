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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuration
st.set_page_config(layout="wide", page_title="Wind Energy Analytics Dashboard", page_icon="🌬️")

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {background-color: #0E1117; color: white;}
    .sidebar .sidebar-content {background-color: #1E1E1E;}
    .stTextInput input, .stSelectbox select, .stSlider div {background-color: #1E1E1E; color: white;}
    .st-bb {background-color: #1E1E1E;}
    .st-at {background-color: #1E1E1E;}
    .metric-card {border-radius: 10px; padding: 15px; background-color: #1E1E1E; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .turbine-card {border-left: 5px solid #4CAF50;}
    .wind-card {border-left: 5px solid #2196F3;}
    h1, h2, h3, h4, h5, h6 {color: white !important;}
    p, div {color: white !important;}
    label {color: white !important;}
    .st-bh, .st-bi, .st-bj, .st-bk {color: white !important;}
    .error-message {color: #FF4B4B; font-weight: bold;}
    .success-message {color: #4CAF50; font-weight: bold;}
    .warning-message {color: #FFA500; font-weight: bold;}
    .info-message {color: #1E90FF; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# API Functions
@st.cache_data(ttl=3600)
def get_coordinates(location):
    """Get coordinates with strict validation"""
    if not location or len(location.strip()) < 2:
        return None, None, "Please enter a valid location name (e.g., 'Chicago, US')", None
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
        headers = {"User-Agent": "WindEnergyApp/1.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return None, None, f"Location '{location}' not found. Please enter a valid city name.", None
            
        # Get the first result with valid coordinates and proper location type
        for item in data:
            try:
                # Only accept cities, towns, villages, etc. - reject ambiguous results
                item_type = item.get('type', '')
                item_class = item.get('class', '')
                
                if item_type not in ['city', 'town', 'village', 'administrative']:
                    continue
                    
                lat = float(item.get('lat', 0))
                lon = float(item.get('lon', 0))
                display_name = item.get('display_name', '')
                
                if (-90 <= lat <= 90 and -180 <= lon <= 180 and 
                    ',' in display_name and len(display_name) > 5):
                    return lat, lon, None, display_name.split(',')[0]
            except (ValueError, TypeError):
                continue
                
        return None, None, f"'{location}' doesn't appear to be a valid city name. Please try again with a proper location.", None
    except Exception as e:
        return None, None, f"API Error: {str(e)}", None

@st.cache_data(ttl=3600)
def get_weather_data(lat, lon):
    """Get weather data with validation"""
    try:
        # Get historical data for past 5 days and forecast for next 2 days (48 hours)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days=5&forecast_days=2"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Validate the response contains required data
        if not all(key in data.get('hourly', {}) for key in ['wind_speed_10m', 'wind_direction_10m', 'temperature_2m']):
            return {"error": "Incomplete weather data received for this location"}
            
        return data
    except Exception as e:
        return {"error": str(e)}

# Turbine Models
class WindTurbine:
    def __init__(self, name, cut_in, rated, cut_out, max_power, rotor_diam):
        self.name = name
        self.cut_in = cut_in
        self.rated = rated
        self.cut_out = cut_out
        self.max_power = max_power
        self.rotor_diam = rotor_diam
    
    def power_output(self, wind_speed):
        wind_speed = np.array(wind_speed)
        power = np.zeros_like(wind_speed)
        mask = (wind_speed >= self.cut_in) & (wind_speed <= self.rated)
        power[mask] = self.max_power * ((wind_speed[mask] - self.cut_in)/(self.rated - self.cut_in))**3
        power[wind_speed > self.rated] = self.max_power
        power[wind_speed > self.cut_out] = 0
        return power

# Turbine Database
TURBINES = {
    "Vestas V80-2.0MW": WindTurbine("Vestas V80-2.0MW", 4, 15, 25, 2000, 80),
    "GE 1.5sle": WindTurbine("GE 1.5sle", 3.5, 14, 25, 1500, 77),
    "Suzlon S88-2.1MW": WindTurbine("Suzlon S88-2.1MW", 3, 12, 25, 2100, 88),
    "Enercon E-53/800": WindTurbine("Enercon E-53/800", 2.5, 13, 25, 800, 53),
    "Custom Turbine": None
}

# Air Density Calculation
def calculate_air_density(temperature, humidity, pressure):
    R_d = 287.05  # Gas constant for dry air (J/kg·K)
    R_v = 461.495  # Gas constant for water vapor (J/kg·K)
    e_s = 0.61121 * np.exp((18.678 - temperature/234.5) * (temperature/(257.14 + temperature)))
    e = (humidity / 100) * e_s
    rho = (pressure * 100) / (R_d * (temperature + 273.15)) * (1 - (0.378 * e) / (pressure * 100))
    return rho

# Weibull Distribution
def weibull(x, k, A):
    return (k/A) * ((x/A)**(k-1)) * np.exp(-(x/A)**k)

# Wind Speed Prediction Model
def train_wind_speed_model(df):
    # Create enhanced features from datetime
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['day_of_year'] = df['Time'].dt.dayofyear
    df['month'] = df['Time'].dt.month
    
    # Add lag features
    df['wind_speed_lag1'] = df['Wind Speed (m/s)'].shift(1)
    df['wind_speed_lag2'] = df['Wind Speed (m/s)'].shift(2)
    df['wind_speed_lag3'] = df['Wind Speed (m/s)'].shift(3)
    
    # Drop rows with NA values from lag features
    df = df.dropna()
    
    # Prepare data
    X = df[['hour_sin', 'hour_cos', 'day_of_week', 'day_of_year', 'month',
            'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)',
            'wind_speed_lag1', 'wind_speed_lag2', 'wind_speed_lag3']]
    y = df['Wind Speed (m/s)']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train enhanced model
    model = RandomForestRegressor(n_estimators=200, 
                                max_depth=10, 
                                min_samples_split=5,
                                random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate test accuracy (R-squared score)
    test_accuracy = model.score(X_test, y_test)
    
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
            'month': time.month,
            'Temperature (°C)': last_data_point['Temperature (°C)'],
            'Humidity (%)': last_data_point['Humidity (%)'],
            'Pressure (hPa)': last_data_point['Pressure (hPa)'],
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

# UI Components
def main():
    st.title("🌬️ Wind Energy Analytics Dashboard")
    st.markdown("**Analyze wind patterns, select optimal turbines, and forecast energy generation**")
    
    with st.expander("⚙️ Configuration Panel", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("📍 Location", "Chennai, India", 
                                   help="Enter a valid city name and country (e.g., 'New York, US')")
            
        with col2:
            turbine_model = st.selectbox("🌀 Turbine Model", list(TURBINES.keys()), index=0)
            if turbine_model == "Custom Turbine":
                st.number_input("Cut-in Speed (m/s)", min_value=1.0, max_value=10.0, value=3.0)
                st.number_input("Rated Speed (m/s)", min_value=5.0, max_value=20.0, value=12.0)
                st.number_input("Cut-out Speed (m/s)", min_value=15.0, max_value=30.0, value=25.0)
                st.number_input("Rated Power (kW)", min_value=100, max_value=10000, value=2000)
    
    if st.button("🚀 Analyze Wind Data"):
        with st.spinner("Fetching wind data and performing analysis..."):
            # Data Acquisition with strict validation
            lat, lon, error, display_name = get_coordinates(location)
            
            if error:
                st.error(f"❌ {error}")
                return
                
            st.success(f"🔍 Location found: {display_name} (Latitude: {lat:.4f}, Longitude: {lon:.4f})")
            
            data = get_weather_data(lat, lon)
            if 'error' in data:
                st.error(f"❌ Weather API Error: {data['error']}")
                return
            
            # Data Processing
            times = pd.to_datetime(data['hourly']['time'])
            df = pd.DataFrame({
                "Time": times,
                "Wind Speed (m/s)": data['hourly']['wind_speed_10m'],
                "Wind Direction": data['hourly']['wind_direction_10m'],
                "Temperature (°C)": data['hourly']['temperature_2m'],
                "Humidity (%)": data['hourly']['relative_humidity_2m'],
                "Pressure (hPa)": data['hourly']['surface_pressure']
            })
            
            # Validate we got actual data
            if df['Wind Speed (m/s)'].isnull().all():
                st.error("❌ No valid wind data received for this location. Please try a different location.")
                return
                
            # Air Density Calculation
            df['Air Density (kg/m³)'] = calculate_air_density(
                df['Temperature (°C)'], 
                df['Humidity (%)'], 
                df['Pressure (hPa)']
            )
            
            # Power Calculation
            turbine = TURBINES[turbine_model]
            df['Power Output (kW)'] = turbine.power_output(df['Wind Speed (m/s)'])
            df['Energy Output (kWh)'] = df['Power Output (kW)']  # Assuming 1 hour intervals
            
            # Split into historical and forecast data
            now = datetime.now()
            historical_df = df[df['Time'] <= now]
            forecast_df = df[df['Time'] > now]
            
            # Train wind speed prediction model
            model, features, test_accuracy = train_wind_speed_model(historical_df.copy())
            
            # Predict future wind speeds (48 hours)
            last_data_point = historical_df.iloc[-1].to_dict()
            future_hours = 48
            future_times, future_wind = predict_future_wind(model, features, last_data_point, future_hours)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Time': future_times,
                'Predicted Wind Speed (m/s)': future_wind,
                'Lower Bound': future_wind * 0.95,
                'Upper Bound': future_wind * 1.05
            })
            
            # Combine historical and predicted data
            combined_df = pd.concat([
                historical_df[['Time', 'Wind Speed (m/s)']].rename(columns={'Wind Speed (m/s)': 'Actual Wind Speed (m/s)'}),
                pred_df[['Time', 'Predicted Wind Speed (m/s)']]
            ])
            
            # Dashboard Layout
            st.success(f"✅ Analysis completed for {display_name}")
            
            # Key Metrics
            st.subheader("📊 Key Performance Indicators")
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Average Wind Speed", f"{df['Wind Speed (m/s)'].mean():.2f} m/s")
            col2.metric("💨 Max Wind Speed", f"{df['Wind Speed (m/s)'].max():.2f} m/s")
            col3.metric("⚡ Total Energy Output", f"{df['Energy Output (kWh)'].sum()/1000:.2f} MWh")
            
            # Main Visualization
            st.subheader("🌪️ Wind Speed Analysis (Historical + Forecast)")
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_df['Time'],
                y=historical_df['Wind Speed (m/s)'],
                name='Historical Data',
                line=dict(color='#1f77b4', width=2),
                mode='lines'
            ))
            
            # Forecast data (actual API forecast)
            fig.add_trace(go.Scatter(
                x=forecast_df['Time'],
                y=forecast_df['Wind Speed (m/s)'],
                name='API Forecast',
                line=dict(color='#ff7f0e', width=2),
                mode='lines'
            ))
            
            # Model predictions
            fig.add_trace(go.Scatter(
                x=pred_df['Time'],
                y=pred_df['Predicted Wind Speed (m/s)'],
                name='Model Prediction',
                line=dict(color='#2ca02c', width=3, dash='dot')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pred_df['Time'],
                y=pred_df['Upper Bound'],
                line=dict(width=0),
                showlegend=False,
                mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=pred_df['Time'],
                y=pred_df['Lower Bound'],
                fill='tonexty',
                fillcolor='rgba(44, 160, 44, 0.2)',
                line=dict(width=0),
                name='Confidence Interval',
                mode='lines'
            ))
            
            # Add vertical line for current time
            fig.add_vline(x=now, line_dash="dash", line_color="red", 
                         annotation_text="Current Time", 
                         annotation_position="top left")
            
            fig.update_layout(
                title="Wind Speed Timeline (Historical Data + Forecast)",
                xaxis_title="Time",
                yaxis_title="Wind Speed (m/s)",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance
            st.subheader("📈 Model Performance")
            
            # Get test predictions
            X_test = historical_df[features]
            historical_df['Predicted Wind Speed (m/s)'] = model.predict(X_test)
            
            # Metrics calculation
            y_test = historical_df['Wind Speed (m/s)']
            y_pred = historical_df['Predicted Wind Speed (m/s)']
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{mae:.2f} m/s")
            col2.metric("Root Mean Squared Error", f"{rmse:.2f} m/s")
            col3.metric("Model Accuracy (R²)", f"{r2:.2f}")
            
            # Turbine performance
            st.subheader("🌀 Turbine Performance")
            
            # Calculate power for both historical and predicted
            historical_df['Power Output (kW)'] = turbine.power_output(historical_df['Wind Speed (m/s)'])
            pred_df['Power Output (kW)'] = turbine.power_output(pred_df['Predicted Wind Speed (m/s)'])
            
            # Combine power data
            power_df = pd.concat([
                historical_df[['Time', 'Power Output (kW)']],
                pred_df[['Time', 'Power Output (kW)']].rename(columns={'Power Output (kW)': 'Predicted Power Output (kW)'})
            ])
            
            fig = px.line(power_df, x="Time", y=["Power Output (kW)", "Predicted Power Output (kW)"], 
                         title="Turbine Power Output", template="plotly_dark")
            fig.add_vline(x=now, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
