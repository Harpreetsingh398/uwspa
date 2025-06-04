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
    .metric-card {border-radius: 10px; padding: 15px; background-color: #1E1E1E; margin: 10px;}
</style>
""", unsafe_allow_html=True)

# API Functions
@st.cache_data(ttl=3600)
def get_coordinates(location):
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
            
        for item in data:
            try:
                lat = float(item.get('lat', 0))
                lon = float(item.get('lon', 0))
                display_name = item.get('display_name', '')
                
                if (-90 <= lat <= 90 and -180 <= lon <= 180 and 
                    ',' in display_name and len(display_name) > 5):
                    return lat, lon, None, display_name.split(',')[0]
            except (ValueError, TypeError):
                continue
                
        return None, None, f"Invalid location format", None
    except Exception as e:
        return None, None, f"API Error: {str(e)}", None

@st.cache_data(ttl=3600)
def get_weather_data(lat, lon, past_days=2, forecast_days=2):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days={past_days}&forecast_days={forecast_days}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not all(key in data.get('hourly', {}) for key in ['wind_speed_10m', 'wind_direction_10m', 'temperature_2m']):
            return {"error": "Incomplete weather data"}
            
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

def calculate_air_density(temperature, humidity, pressure):
    R_d = 287.05
    R_v = 461.495
    e_s = 0.61121 * np.exp((18.678 - temperature/234.5) * (temperature/(257.14 + temperature)))
    e = (humidity / 100) * e_s
    rho = (pressure * 100) / (R_d * (temperature + 273.15)) * (1 - (0.378 * e) / (pressure * 100))
    return rho

def weibull(x, k, A):
    return (k/A) * ((x/A)**(k-1)) * np.exp(-(x/A)**k)

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
    
    pred_df = pd.DataFrame(pred_data)[features]
    predictions = model.predict(pred_df)
    return future_times, predictions

def main():
    st.title("🌬️ Wind Energy Analytics Dashboard")
    
    with st.expander("⚙️ Configuration Panel", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("📍 Location", "Chicago, US")
            turbine_model = st.selectbox("🌀 Turbine Model", list(TURBINES.keys()), index=0)
        with col2:
            past_days = st.slider("📅 Past Days for Training", 1, 7, 2)
            future_hours = st.slider("⏳ Hours to Forecast", 6, 48, 24, step=6)
    
    if st.button("🚀 Analyze Wind Data"):
        with st.spinner("Fetching and analyzing data..."):
            lat, lon, error, display_name = get_coordinates(location)
            
            if error:
                st.error(f"❌ {error}")
                return
                
            st.success(f"🔍 Location: {display_name} (Lat: {lat:.4f}, Lon: {lon:.4f})")
            
            data = get_weather_data(lat, lon, past_days, 2)
            if 'error' in data:
                st.error(f"❌ Weather API Error: {data['error']}")
                return
            
            # Process data
            times = pd.to_datetime(data['hourly']['time'])
            df = pd.DataFrame({
                "Time": times,
                "Wind Speed (m/s)": data['hourly']['wind_speed_10m'],
                "Wind Direction": data['hourly']['wind_direction_10m'],
                "Temperature (°C)": data['hourly']['temperature_2m'],
                "Humidity (%)": data['hourly']['relative_humidity_2m'],
                "Pressure (hPa)": data['hourly']['surface_pressure']
            })
            
            # Calculate air density
            df['Air Density (kg/m³)'] = calculate_air_density(
                df['Temperature (°C)'], 
                df['Humidity (%)'], 
                df['Pressure (hPa)']
            )
            
            # Power calculation
            turbine = TURBINES[turbine_model]
            df['Power Output (kW)'] = turbine.power_output(df['Wind Speed (m/s)'])
            
            # Split into historical and forecast data
            now = datetime.now()
            historical_df = df[df['Time'] < now]
            forecast_df = df[df['Time'] >= now]
            
            # Train model on historical data
            model, features, test_accuracy = train_wind_speed_model(historical_df.copy())
            
            # Make predictions
            last_data_point = historical_df.iloc[-1].to_dict()
            pred_times, pred_wind = predict_future_wind(model, features, last_data_point, future_hours)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Time': pred_times,
                'Predicted Wind Speed (m/s)': pred_wind,
                'Lower Bound': pred_wind * 0.95,
                'Upper Bound': pred_wind * 1.05
            })
            
            # Combine all data for visualization
            historical_df['Type'] = 'Historical'
            forecast_df['Type'] = 'Forecast'
            pred_df['Type'] = 'Predicted'
            
            # Main visualization
            st.subheader("🌪️ Wind Speed Timeline")
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_df['Time'],
                y=historical_df['Wind Speed (m/s)'],
                name='Historical Data',
                line=dict(color='#1f77b4', width=2),
                mode='lines'
            ))
            
            # Forecast data (if available)
            if not forecast_df.empty:
                fig.add_trace(go.Scatter(
                    x=forecast_df['Time'],
                    y=forecast_df['Wind Speed (m/s)'],
                    name='API Forecast',
                    line=dict(color='#2ca02c', width=2),
                    mode='lines'
                ))
            
            # Model predictions
            fig.add_trace(go.Scatter(
                x=pred_df['Time'],
                y=pred_df['Predicted Wind Speed (m/s)'],
                name='Model Prediction',
                line=dict(color='#ff7f0e', width=3)
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
                fillcolor='rgba(255,127,14,0.2)',
                line=dict(width=0),
                name='Confidence Interval',
                mode='lines'
            ))
            
            # Add vertical line for current time
            fig.add_vline(x=now, line_dash="dash", line_color="red", 
                          annotation_text="Now", annotation_position="top left")
            
            fig.update_layout(
                title=f"Wind Speed Timeline - Past {past_days} Days and Next {future_hours} Hours",
                xaxis_title="Time",
                yaxis_title="Wind Speed (m/s)",
                template="plotly_dark",
                hovermode="x unified",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.subheader("📊 Key Metrics")
            cols = st.columns(4)
            cols[0].metric("Historical Avg Speed", f"{historical_df['Wind Speed (m/s)'].mean():.2f} m/s")
            cols[1].metric("Forecast Avg Speed", f"{forecast_df['Wind Speed (m/s)'].mean():.2f} m/s" if not forecast_df.empty else "N/A")
            cols[2].metric("Predicted Avg Speed", f"{pred_df['Predicted Wind Speed (m/s)'].mean():.2f} m/s")
            cols[3].metric("Model Accuracy", f"{test_accuracy:.2%}")
            
            # Power output visualization
            st.subheader("⚡ Power Output")
            
            # Calculate power for predictions
            pred_df['Power Output (kW)'] = turbine.power_output(pred_df['Predicted Wind Speed (m/s)'])
            
            power_fig = go.Figure()
            
            if not historical_df.empty:
                power_fig.add_trace(go.Scatter(
                    x=historical_df['Time'],
                    y=historical_df['Power Output (kW)'],
                    name='Historical Power',
                    line=dict(color='#1f77b4', width=2),
                    mode='lines'
                ))
            
            if not forecast_df.empty:
                power_fig.add_trace(go.Scatter(
                    x=forecast_df['Time'],
                    y=forecast_df['Power Output (kW)'],
                    name='Forecast Power',
                    line=dict(color='#2ca02c', width=2),
                    mode='lines'
                ))
            
            power_fig.add_trace(go.Scatter(
                x=pred_df['Time'],
                y=pred_df['Power Output (kW)'],
                name='Predicted Power',
                line=dict(color='#ff7f0e', width=3),
                mode='lines'
            ))
            
            power_fig.add_vline(x=now, line_dash="dash", line_color="red")
            
            power_fig.update_layout(
                title=f"Power Output - {turbine_model}",
                xaxis_title="Time",
                yaxis_title="Power Output (kW)",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(power_fig, use_container_width=True)

if __name__ == "__main__":
    main()
