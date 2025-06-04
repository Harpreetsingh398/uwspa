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

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0E1117; color: white;}
    .sidebar .sidebar-content {background-color: #1E1E1E;}
    .stTextInput input, .stSelectbox select, .stSlider div {background-color: #1E1E1E; color: white;}
    h1, h2, h3, h4, h5, h6 {color: white !important;}
    .metric-card {border-radius: 10px; padding: 15px; background-color: #1E1E1E; margin: 5px;}
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
def get_weather_data(lat, lon, past_days=2, forecast_days=2):
    """Get weather data with both past and forecast"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days={past_days}&forecast_days={forecast_days}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not all(key in data.get('hourly', {}) for key in ['wind_speed_10m', 'wind_direction_10m']):
            return {"error": "Incomplete weather data received"}
            
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
    "Enercon E-53/800": WindTurbine("Enercon E-53/800", 2.5, 13, 25, 800, 53)
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
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['day_of_year'] = df['Time'].dt.dayofyear
    df['month'] = df['Time'].dt.month
    
    df['wind_speed_lag1'] = df['Wind Speed (m/s)'].shift(1)
    df['wind_speed_lag2'] = df['Wind Speed (m/s)'].shift(2)
    df['wind_speed_lag3'] = df['Wind Speed (m/s)'].shift(3)
    
    df = df.dropna()
    
    X = df[['hour_sin', 'hour_cos', 'day_of_week', 'day_of_year', 'month',
            'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)',
            'wind_speed_lag1', 'wind_speed_lag2', 'wind_speed_lag3']]
    y = df['Wind Speed (m/s)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    test_accuracy = model.score(X_test, y_test)
    
    return model, X.columns.tolist(), test_accuracy

def predict_future_wind(model, features, last_data_point, future_hours):
    future_times = [last_data_point['Time'] + timedelta(hours=i) for i in range(1, future_hours+1)]
    
    pred_data = []
    for i, time in enumerate(future_times):
        lag1 = last_data_point['Wind Speed (m/s)']
        lag2 = last_data_point.get('wind_speed_lag1', lag1)
        lag3 = last_data_point.get('wind_speed_lag2', lag2)
        
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
    
    pred_df = pd.DataFrame(pred_data)[features]
    predictions = model.predict(pred_df)
    
    return future_times, predictions

def main():
    st.title("🌬️ Wind Energy Analytics Dashboard")
    
    with st.expander("⚙️ Configuration Panel", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("📍 Location", "Chennai, India")
            turbine_model = st.selectbox("🌀 Turbine Model", list(TURBINES.keys()), index=0)
        with col2:
            past_days = st.slider("📅 Past Days for Training", 1, 7, 2)
            future_hours = st.slider("🔮 Hours to Forecast", 6, 48, 24, step=6)
    
    if st.button("🚀 Analyze Wind Data"):
        with st.spinner("Fetching data and performing analysis..."):
            # Get coordinates
            lat, lon, error, display_name = get_coordinates(location)
            if error:
                st.error(f"❌ {error}")
                return
                
            st.success(f"🔍 Location found: {display_name} (Latitude: {lat:.4f}, Longitude: {lon:.4f})")
            
            # Get weather data (past + forecast)
            data = get_weather_data(lat, lon, past_days=past_days, forecast_days=2)
            if 'error' in data:
                st.error(f"❌ Weather API Error: {data['error']}")
                return
            
            # Process data
            hourly = data['hourly']
            df = pd.DataFrame({
                "Time": pd.to_datetime(hourly['time']),
                "Wind Speed (m/s)": hourly['wind_speed_10m'],
                "Wind Direction": hourly['wind_direction_10m'],
                "Temperature (°C)": hourly['temperature_2m'],
                "Humidity (%)": hourly['relative_humidity_2m'],
                "Pressure (hPa)": hourly['surface_pressure']
            })
            
            # Calculate air density
            df['Air Density (kg/m³)'] = calculate_air_density(
                df['Temperature (°C)'], 
                df['Humidity (%)'], 
                df['Pressure (hPa)']
            )
            
            # Calculate power output
            turbine = TURBINES[turbine_model]
            df['Power Output (kW)'] = turbine.power_output(df['Wind Speed (m/s)'])
            
            # Split into past and forecast data
            now = datetime.now()
            past_df = df[df['Time'] < now]
            forecast_df = df[df['Time'] >= now]
            
            # Train model on past data
            model, features, test_accuracy = train_wind_speed_model(past_df.copy())
            
            # Make predictions for future
            if len(past_df) > 0:
                last_point = past_df.iloc[-1].to_dict()
                pred_times, pred_speeds = predict_future_wind(model, features, last_point, future_hours)
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    'Time': pred_times,
                    'Predicted Wind Speed (m/s)': pred_speeds,
                    'Lower Bound': pred_speeds * 0.95,
                    'Upper Bound': pred_speeds * 1.05
                })
                
                # Combine with actual forecast
                combined_forecast = pd.concat([
                    forecast_df[['Time', 'Wind Speed (m/s)']].rename(columns={'Wind Speed (m/s)': 'Actual Wind Speed (m/s)'}),
                    pred_df[['Time', 'Predicted Wind Speed (m/s)']]
                ])
            
            # Display key metrics
            st.subheader("📊 Key Metrics")
            cols = st.columns(4)
            cols[0].metric("🌡️ Avg Wind Speed", f"{df['Wind Speed (m/s)'].mean():.2f} m/s")
            cols[1].metric("💨 Max Wind Speed", f"{df['Wind Speed (m/s)'].max():.2f} m/s")
            cols[2].metric("⚡ Max Power", f"{df['Power Output (kW)'].max():.0f} kW")
            cols[3].metric("🌀 Predominant Dir", f"{df['Wind Direction'].mode()[0]}°")
            
            # Main visualization
            st.subheader("🌪️ Wind Speed Analysis & Forecast")
            
            fig = go.Figure()
            
            # Past data
            if len(past_df) > 0:
                fig.add_trace(go.Scatter(
                    x=past_df['Time'], 
                    y=past_df['Wind Speed (m/s)'], 
                    name='Historical Data',
                    line=dict(color='#1f77b4', width=2)
                ))
            
            # API Forecast
            if len(forecast_df) > 0:
                fig.add_trace(go.Scatter(
                    x=forecast_df['Time'],
                    y=forecast_df['Wind Speed (m/s)'],
                    name='API Forecast',
                    line=dict(color='#2ca02c', width=2)
                ))
            
            # Model Prediction
            if len(past_df) > 0:
                fig.add_trace(go.Scatter(
                    x=pred_df['Time'],
                    y=pred_df['Predicted Wind Speed (m/s)'],
                    name='Model Prediction',
                    line=dict(color='#ff7f0e', width=3)
                ))
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
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(width=0),
                    name='Confidence Interval',
                    mode='lines'
                ))
            
            fig.update_layout(
                title=f"Wind Speed Timeline for {display_name}",
                xaxis_title="Date & Time",
                yaxis_title="Wind Speed (m/s)",
                template="plotly_dark",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance
            st.subheader("📈 Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy (R²)", f"{test_accuracy:.2%}")
            
            if len(past_df) > 0 and len(forecast_df) > 0:
                # Compare API forecast vs model prediction
                comparison_df = pd.merge(
                    forecast_df[['Time', 'Wind Speed (m/s)']].rename(columns={'Wind Speed (m/s)': 'API_Forecast'}),
                    pred_df[['Time', 'Predicted Wind Speed (m/s)']].rename(columns={'Predicted Wind Speed (m/s)': 'Model_Prediction'}),
                    on='Time',
                    how='inner'
                )
                
                if not comparison_df.empty:
                    mae = mean_absolute_error(comparison_df['API_Forecast'], comparison_df['Model_Prediction'])
                    with col2:
                        st.metric("MAE vs API Forecast", f"{mae:.2f} m/s")
                    
                    fig2 = px.scatter(
                        comparison_df,
                        x='API_Forecast',
                        y='Model_Prediction',
                        title="Model Prediction vs API Forecast",
                        labels={
                            'API_Forecast': 'API Forecast (m/s)',
                            'Model_Prediction': 'Model Prediction (m/s)'
                        },
                        template="plotly_dark"
                    )
                    fig2.add_shape(
                        type="line",
                        x0=0, y0=0,
                        x1=max(comparison_df['API_Forecast'].max(), comparison_df['Model_Prediction'].max()),
                        y1=max(comparison_df['API_Forecast'].max(), comparison_df['Model_Prediction'].max()),
                        line=dict(color="#00CC96", width=3, dash="dash")
                    )
                    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
