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
st.set_page_config(layout="wide", page_title="Wind Energy Dashboard", page_icon="🌬️")

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {background-color: #0E1117; color: white;}
    .stTextInput input, .stSelectbox select {background-color: #1E1E1E; color: white;}
    h1, h2, h3, h4, h5, h6 {color: white !important;}
    .error-message {color: #FF4B4B; font-weight: bold;}
    .success-message {color: #4CAF50; font-weight: bold;}
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
        display_name = first.get('display_name', '').split(',')[0]
        
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon, None, display_name
        return None, None, "Invalid coordinates received", None
    except Exception as e:
        return None, None, f"API Error: {str(e)}", None

@st.cache_data(ttl=3600)
def get_weather_data(lat, lon):
    """Get weather data for past 5 days and forecast"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days=5"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Turbine Model
class WindTurbine:
    def __init__(self, cut_in=3.5, rated=14, cut_out=25, max_power=1500):
        self.cut_in = cut_in
        self.rated = rated
        self.cut_out = cut_out
        self.max_power = max_power
    
    def power_output(self, wind_speed):
        wind_speed = np.array(wind_speed)
        power = np.zeros_like(wind_speed)
        mask = (wind_speed >= self.cut_in) & (wind_speed <= self.rated)
        power[mask] = self.max_power * ((wind_speed[mask] - self.cut_in)/(self.rated - self.cut_in))**3
        power[wind_speed > self.rated] = self.max_power
        power[wind_speed > self.cut_out] = 0
        return power

def prepare_data(data):
    """Prepare historical and forecast data"""
    times = pd.to_datetime(data['hourly']['time'])
    df = pd.DataFrame({
        "Time": times,
        "Wind Speed (m/s)": data['hourly']['wind_speed_10m'],
        "Wind Direction": data['hourly']['wind_direction_10m'],
        "Temperature (°C)": data['hourly']['temperature_2m'],
        "Humidity (%)": data['hourly']['relative_humidity_2m'],
        "Pressure (hPa)": data['hourly']['surface_pressure']
    })
    
    # Add air density
    df['Air Density (kg/m³)'] = calculate_air_density(
        df['Temperature (°C)'], 
        df['Humidity (%)'], 
        df['Pressure (hPa)']
    )
    
    # Split into historical and forecast
    now = datetime.now()
    historical = df[df['Time'] < now].copy()
    forecast = df[df['Time'] >= now].copy()
    
    return historical, forecast

def calculate_air_density(temp, humidity, pressure):
    """Calculate air density"""
    R_d = 287.05  # Gas constant for dry air
    R_v = 461.495  # Gas constant for water vapor
    e_s = 0.61121 * np.exp((18.678 - temp/234.5) * (temp/(257.14 + temp)))
    e = (humidity / 100) * e_s
    rho = (pressure * 100) / (R_d * (temp + 273.15)) * (1 - (0.378 * e) / (pressure * 100))
    return rho

def train_model(df):
    """Train wind speed prediction model"""
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_year'] = df['Time'].dt.dayofyear
    
    # Lag features
    for i in range(1, 4):
        df[f'lag_{i}'] = df['Wind Speed (m/s)'].shift(i)
    
    df = df.dropna()
    
    features = ['hour_sin', 'hour_cos', 'day_of_year', 'Temperature (°C)', 
               'Humidity (%)', 'Pressure (hPa)', 'lag_1', 'lag_2', 'lag_3']
    X = df[features]
    y = df['Wind Speed (m/s)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, features

def predict_future(model, features, last_point, hours=48):
    """Predict future wind speeds"""
    predictions = []
    times = []
    current = last_point.copy()
    
    for i in range(1, hours+1):
        # Create features for next time step
        next_time = current['Time'] + timedelta(hours=1)
        hour = next_time.hour
        day_of_year = next_time.timetuple().tm_yday
        
        features_dict = {
            'hour_sin': np.sin(2 * np.pi * hour/24),
            'hour_cos': np.cos(2 * np.pi * hour/24),
            'day_of_year': day_of_year,
            'Temperature (°C)': current['Temperature (°C)'],
            'Humidity (%)': current['Humidity (%)'],
            'Pressure (hPa)': current['Pressure (hPa)'],
            'lag_1': current['Wind Speed (m/s)'],
            'lag_2': current.get('lag_1', current['Wind Speed (m/s)']),
            'lag_3': current.get('lag_2', current['Wind Speed (m/s)'])
        }
        
        # Predict next wind speed
        X_pred = pd.DataFrame([features_dict])[features]
        pred = model.predict(X_pred)[0]
        
        # Update current state
        current['Time'] = next_time
        current['lag_3'] = current.get('lag_2', current['Wind Speed (m/s)'])
        current['lag_2'] = current.get('lag_1', current['Wind Speed (m/s)'])
        current['lag_1'] = current['Wind Speed (m/s)']
        current['Wind Speed (m/s)'] = pred
        
        predictions.append(pred)
        times.append(next_time)
    
    return times, predictions

def main():
    st.title("🌬️ Wind Energy Analytics Dashboard")
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Location", "New York, US")
    with col2:
        turbine_power = st.selectbox("Turbine Power (kW)", [1500, 2000, 2500], index=0)
    
    if st.button("Analyze"):
        with st.spinner("Fetching data and analyzing..."):
            # Get location coordinates
            lat, lon, error, display_name = get_coordinates(location)
            if error:
                st.error(f"Error: {error}")
                return
                
            st.success(f"Location: {display_name} (Lat: {lat:.2f}, Lon: {lon:.2f})")
            
            # Get weather data
            data = get_weather_data(lat, lon)
            if 'error' in data:
                st.error(f"Weather API Error: {data['error']}")
                return
                
            # Prepare data
            historical, forecast = prepare_data(data)
            
            # Train model
            model, features = train_model(historical)
            
            # Get last historical point for prediction
            last_point = historical.iloc[-1].to_dict()
            
            # Predict next 48 hours
            pred_times, pred_speeds = predict_future(model, features, last_point)
            
            # Create prediction DataFrame
            pred_df = pd.DataFrame({
                'Time': pred_times,
                'Predicted Wind Speed (m/s)': pred_speeds,
                'Type': 'Prediction'
            })
            
            # Prepare historical DataFrame
            hist_df = historical[['Time', 'Wind Speed (m/s)']].copy()
            hist_df['Type'] = 'Historical'
            
            # Combine data for plotting
            plot_df = pd.concat([
                hist_df.rename(columns={'Wind Speed (m/s)': 'Speed (m/s)'}),
                pred_df.rename(columns={'Predicted Wind Speed (m/s)': 'Speed (m/s)'})
            ])
            
            # Calculate power output
            turbine = WindTurbine(max_power=turbine_power)
            plot_df['Power (kW)'] = turbine.power_output(plot_df['Speed (m/s)'])
            
            # Visualization
            st.subheader("Wind Speed and Power Forecast (48 hours)")
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=plot_df[plot_df['Type'] == 'Historical']['Time'],
                y=plot_df[plot_df['Type'] == 'Historical']['Speed (m/s)'],
                name='Historical Wind Speed',
                line=dict(color='blue')
            ))
            
            # Predicted data
            fig.add_trace(go.Scatter(
                x=plot_df[plot_df['Type'] == 'Prediction']['Time'],
                y=plot_df[plot_df['Type'] == 'Prediction']['Speed (m/s)'],
                name='Predicted Wind Speed',
                line=dict(color='red', dash='dot')
            ))
            
            # Power output on secondary y-axis
            fig.add_trace(go.Scatter(
                x=plot_df['Time'],
                y=plot_df['Power (kW)'],
                name='Power Output',
                yaxis='y2',
                line=dict(color='green')
            ))
            
            # Add vertical line for current time
            now = datetime.now()
            fig.add_vline(x=now, line_dash="dash", line_color="white")
            
            fig.update_layout(
                title='Wind Speed and Power Output',
                xaxis_title='Time',
                yaxis_title='Wind Speed (m/s)',
                yaxis2=dict(
                    title='Power (kW)',
                    overlaying='y',
                    side='right'
                ),
                template="plotly_dark",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            # Historical stats
            avg_speed = historical['Wind Speed (m/s)'].mean()
            max_speed = historical['Wind Speed (m/s)'].max()
            col1.metric("Historical Avg Wind Speed", f"{avg_speed:.1f} m/s")
            col2.metric("Historical Max Wind Speed", f"{max_speed:.1f} m/s")
            
            # Predicted stats
            pred_avg = pred_df['Predicted Wind Speed (m/s)'].mean()
            pred_max = pred_df['Predicted Wind Speed (m/s)'].max()
            col3.metric("Predicted Avg Wind Speed", f"{pred_avg:.1f} m/s")
            
            # Energy calculations
            total_energy = plot_df['Power (kW)'].sum() / 1000  # MWh
            capacity_factor = (plot_df['Power (kW)'].mean() / turbine_power) * 100
            col1.metric("Total Energy (48h)", f"{total_energy:.1f} MWh")
            col2.metric("Capacity Factor", f"{capacity_factor:.1f}%")
            
            # Model evaluation
            y_test = historical['Wind Speed (m/s)']
            X_test = historical[features]
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            col3.metric("Model R² Score", f"{r2:.2f}")

if __name__ == "__main__":
    main()
