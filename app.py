import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import weibull_min
from windrose import WindroseAxes
from io import BytesIO
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(layout="wide", page_title="Wind Energy Dashboard")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: white; color: #333;}
    .stTextInput input, .stSelectbox select {border: 1px solid #ddd;}
    .metric-card {border-radius: 5px; padding: 15px; background-color: #f9f9f9; margin-bottom: 10px;}
    .turbine-card {border-left: 4px solid #4CAF50;}
    .wind-card {border-left: 4px solid #2196F3;}
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
        # Historical data (past 5 days)
        hist_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days=5"
        hist_response = requests.get(hist_url)
        hist_response.raise_for_status()
        hist_data = hist_response.json()
        
        # Forecast data (next 48 hours)
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&forecast_days=2"
        forecast_response = requests.get(forecast_url)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        # Combine data
        combined_data = {
            'hourly': {
                'time': hist_data['hourly']['time'] + forecast_data['hourly']['time'],
                'wind_speed_10m': hist_data['hourly']['wind_speed_10m'] + forecast_data['hourly']['wind_speed_10m'],
                'wind_direction_10m': hist_data['hourly']['wind_direction_10m'] + forecast_data['hourly']['wind_direction_10m'],
                'temperature_2m': hist_data['hourly']['temperature_2m'] + forecast_data['hourly']['temperature_2m'],
                'relative_humidity_2m': hist_data['hourly']['relative_humidity_2m'] + forecast_data['hourly']['relative_humidity_2m'],
                'surface_pressure': hist_data['hourly']['surface_pressure'] + forecast_data['hourly']['surface_pressure']
            }
        }
        
        return combined_data
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
    "Suzlon S88-2.1MW": WindTurbine("Suzlon S88-2.1MW", 3, 12, 25, 2100, 88)
}

# Air Density Calculation
def calculate_air_density(temperature, humidity, pressure):
    R_d = 287.05  # Gas constant for dry air (J/kg·K)
    R_v = 461.495  # Gas constant for water vapor (J/kg·K)
    e_s = 0.61121 * np.exp((18.678 - temperature/234.5) * (temperature/(257.14 + temperature)))
    e = (humidity / 100) * e_s
    rho = (pressure * 100) / (R_d * (temperature + 273.15)) * (1 - (0.378 * e) / (pressure * 100))
    return rho

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
            'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)',
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
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    return model, X.columns.tolist(), test_mae, test_r2

def predict_future_wind(model, features, last_data_point, future_hours):
    # Create future timestamps
    future_times = [last_data_point['Time'] + timedelta(hours=i) for i in range(1, future_hours+1)]
    
    # Prepare prediction data
    pred_data = []
    for i, time in enumerate(future_times):
        # Use the last known values
        lag1 = last_data_point['Wind Speed (m/s)']
        lag2 = last_data_point['wind_speed_lag1'] if 'wind_speed_lag1' in last_data_point else lag1
        lag3 = last_data_point['wind_speed_lag2'] if 'wind_speed_lag2' in last_data_point else lag2
        
        row = {
            'hour_sin': np.sin(2 * np.pi * time.hour/24),
            'hour_cos': np.cos(2 * np.pi * time.hour/24),
            'day_of_week': time.weekday(),
            'day_of_year': time.timetuple().tm_yday,
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

def plot_wind_rose(wind_speeds, wind_directions):
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(wind_directions, wind_speeds, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.close()
    return fig

def main():
    st.title("🌬️ Wind Energy Analytics Dashboard")
    
    # User Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.text_input("📍 Location", "Chicago, US")
    with col2:
        turbine_model = st.selectbox("🌀 Turbine Model", list(TURBINES.keys()))
    with col3:
        future_hours = st.slider("Hours to forecast", 6, 48, 24, step=6)
    
    if st.button("Generate Analysis"):
        with st.spinner("Fetching data and performing analysis..."):
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
            
            # Split into historical and forecast data
            now = datetime.now()
            historical_df = df[df['Time'] < now].copy()
            forecast_df = df[df['Time'] >= now].copy()
            
            # Train model on historical data
            model, features, test_mae, test_r2 = train_wind_speed_model(historical_df.copy())
            
            # Make predictions for future
            last_data_point = historical_df.iloc[-1].to_dict()
            future_times, future_wind = predict_future_wind(model, features, last_data_point, future_hours)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Time': future_times,
                'Predicted Wind Speed (m/s)': future_wind,
                'Type': 'Prediction'
            })
            
            # Select turbine
            turbine = TURBINES[turbine_model]
            
            # Calculate power output for all data
            historical_df['Power Output (kW)'] = turbine.power_output(historical_df['Wind Speed (m/s)'])
            forecast_df['Power Output (kW)'] = turbine.power_output(forecast_df['Wind Speed (m/s)'])
            pred_df['Power Output (kW)'] = turbine.power_output(pred_df['Predicted Wind Speed (m/s)'])
            
            # Combine all data for visualization
            historical_df['Type'] = 'Historical'
            forecast_df['Type'] = 'API Forecast'
            
            # Limit forecast data to 48 hours
            forecast_df = forecast_df[forecast_df['Time'] <= now + timedelta(hours=48)]
            
            # Main Wind Speed Forecast Chart
            st.subheader(f"Wind Speed Forecast for {display_name}")
            
            fig1 = go.Figure()
            
            # Historical data
            fig1.add_trace(go.Scatter(
                x=historical_df['Time'],
                y=historical_df['Wind Speed (m/s)'],
                name='Historical Data',
                line=dict(color='#636EFA'),
                mode='lines'
            ))
            
            # API Forecast
            fig1.add_trace(go.Scatter(
                x=forecast_df['Time'],
                y=forecast_df['Wind Speed (m/s)'],
                name='API Forecast',
                line=dict(color='#00CC96'),
                mode='lines'
            ))
            
            # Model Predictions
            fig1.add_trace(go.Scatter(
                x=pred_df['Time'],
                y=pred_df['Predicted Wind Speed (m/s)'],
                name='Model Prediction',
                line=dict(color='#EF553B', dash='dot'),
                mode='lines'
            ))
            
            # Add vertical line for current time
            fig1.add_vline(
                x=now.timestamp() * 1000,
                line_dash="dash",
                line_color="gray",
                annotation_text="Now",
                annotation_position="top left"
            )
            
            # Update layout
            fig1.update_layout(
                xaxis_title="Date/Time",
                yaxis_title="Wind Speed (m/s)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=500
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Key metrics
            st.subheader("Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            # Current wind speed
            current_speed = historical_df.iloc[-1]['Wind Speed (m/s)']
            col1.metric("Current Wind Speed", f"{current_speed:.1f} m/s")
            
            # Average predicted speed
            avg_pred = pred_df['Predicted Wind Speed (m/s)'].mean()
            col2.metric("Avg Predicted Speed", f"{avg_pred:.1f} m/s")
            
            # Model accuracy
            col3.metric("Model Accuracy (R²)", f"{test_r2:.2f}")
            
            # Capacity factor
            total_hours = len(historical_df) + len(forecast_df) + len(pred_df)
            capacity_factor = (historical_df['Power Output (kW)'].sum() + 
                             forecast_df['Power Output (kW)'].sum() + 
                             pred_df['Power Output (kW)'].sum()) / (turbine.max_power * total_hours) * 100
            col4.metric("Capacity Factor", f"{capacity_factor:.1f}%")
            
            # Additional Insights
            st.subheader("Additional Insights")
            
            # Row 1: Power Output and Wind Rose
            col1, col2 = st.columns(2)
            
            with col1:
                # Power Output Chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=historical_df['Time'],
                    y=historical_df['Power Output (kW)'],
                    name='Historical',
                    line=dict(color='#636EFA')
                ))
                fig2.add_trace(go.Scatter(
                    x=forecast_df['Time'],
                    y=forecast_df['Power Output (kW)'],
                    name='API Forecast',
                    line=dict(color='#00CC96')
                ))
                fig2.add_trace(go.Scatter(
                    x=pred_df['Time'],
                    y=pred_df['Power Output (kW)'],
                    name='Prediction',
                    line=dict(color='#EF553B', dash='dot')
                ))
                fig2.add_vline(
                    x=now.timestamp() * 1000,
                    line_dash="dash",
                    line_color="gray"
                )
                fig2.update_layout(
                    title=f"{turbine_model} Power Output",
                    xaxis_title="Date/Time",
                    yaxis_title="Power Output (kW)",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Wind Rose
                st.markdown("**Wind Direction Distribution**")
                wind_rose_fig = plot_wind_rose(
                    historical_df['Wind Speed (m/s)'],
                    historical_df['Wind Direction']
                )
                st.pyplot(wind_rose_fig)
            
            # Row 2: Weibull Distribution and Power Curve
            col1, col2 = st.columns(2)
            
            with col1:
                # Weibull Distribution
                shape, loc, scale = weibull_min.fit(historical_df['Wind Speed (m/s)'], floc=0)
                x = np.linspace(0, historical_df['Wind Speed (m/s)'].max()*1.2, 100)
                pdf = weibull_min.pdf(x, shape, loc, scale)
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=x,
                    y=pdf,
                    name='Weibull Fit',
                    line=dict(color='#FFA15A')
                )
                fig3.add_trace(go.Histogram(
                    x=historical_df['Wind Speed (m/s)'],
                    histnorm='probability density',
                    name='Actual Data',
                    opacity=0.5,
                    marker_color='#636EFA'
                ))
                fig3.update_layout(
                    title=f"Weibull Distribution (k={shape:.2f}, A={scale:.2f})",
                    xaxis_title="Wind Speed (m/s)",
                    yaxis_title="Probability Density",
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Power Curve
                wind_range = np.linspace(0, turbine.cut_out*1.2, 100)
                power_curve = turbine.power_output(wind_range)
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=wind_range,
                    y=power_curve,
                    name='Power Curve',
                    line=dict(color='#19D3F3')
                )
                fig4.add_vline(
                    x=turbine.cut_in,
                    line_dash="dash",
                    annotation_text=f"Cut-in: {turbine.cut_in}m/s",
                    line_color="green"
                )
                fig4.add_vline(
                    x=turbine.rated,
                    line_dash="dash",
                    annotation_text=f"Rated: {turbine.rated}m/s",
                    line_color="blue"
                )
                fig4.add_vline(
                    x=turbine.cut_out,
                    line_dash="dash",
                    annotation_text=f"Cut-out: {turbine.cut_out}m/s",
                    line_color="red"
                )
                fig4.update_layout(
                    title=f"{turbine_model} Power Curve",
                    xaxis_title="Wind Speed (m/s)",
                    yaxis_title="Power Output (kW)",
                    height=400
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            # Row 3: Daily Pattern and Air Density Impact
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily Pattern
                historical_df['Hour'] = historical_df['Time'].dt.hour
                hourly_avg = historical_df.groupby('Hour').agg({
                    'Wind Speed (m/s)': 'mean',
                    'Power Output (kW)': 'mean'
                }).reset_index()
                
                fig5 = go.Figure()
                fig5.add_trace(go.Bar(
                    x=hourly_avg['Hour'],
                    y=hourly_avg['Power Output (kW)'],
                    name='Power Output',
                    marker_color='#AB63FA'
                ))
                fig5.add_trace(go.Scatter(
                    x=hourly_avg['Hour'],
                    y=hourly_avg['Wind Speed (m/s)'],
                    name='Wind Speed',
                    yaxis="y2",
                    line=dict(color='#FFA15A')
                ))
                fig5.update_layout(
                    title="Daily Generation Pattern",
                    xaxis_title="Hour of Day",
                    yaxis_title="Power Output (kW)",
                    yaxis2=dict(
                        title="Wind Speed (m/s)",
                        overlaying="y",
                        side="right"
                    ),
                    height=400
                )
                st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                # Air Density Impact
                fig6 = px.scatter(
                    historical_df,
                    x='Air Density (kg/m³)',
                    y='Power Output (kW)',
                    trendline="ols",
                    title="Air Density Impact on Power Output",
                    labels={
                        'Air Density (kg/m³)': 'Air Density (kg/m³)',
                        'Power Output (kW)': 'Power Output (kW)'
                    }
                )
                fig6.update_traces(
                    marker=dict(size=8, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')),
                    selector=dict(mode='markers')
                )
                fig6.update_layout(height=400)
                st.plotly_chart(fig6, use_container_width=True)

if __name__ == "__main__":
    main()
