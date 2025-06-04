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
st.set_page_config(layout="wide", page_title="Wind Energy Forecast", page_icon="🌬️")

# Custom CSS for clean layout
st.markdown("""
<style>
    .main {background-color: #0E1117; color: white;}
    .stTextInput input, .stSelectbox select {background-color: #1E1E1E; color: white;}
    h1, h2, h3, h4, h5, h6 {color: white !important;}
    .st-bb {background-color: #1E1E1E;}
    .st-at {background-color: #1E1E1E;}
    .metric-card {border-radius: 5px; padding: 10px; background-color: #1E1E1E;}
    .turbine-card {border-left: 3px solid #4CAF50;}
    .wind-card {border-left: 3px solid #2196F3;}
</style>
""", unsafe_allow_html=True)

# API Functions
@st.cache_data(ttl=3600)
def get_coordinates(location):
    """Get coordinates with validation"""
    if not location or len(location.strip()) < 2:
        return None, None, "Please enter a valid location name (e.g., 'Chicago, US')", None
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
        headers = {"User-Agent": "WindEnergyApp/1.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return None, None, f"Location '{location}' not found.", None
            
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
    """Get weather data for past 5 days and next 48 hours"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days=5&forecast_days=2"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Turbine Model
class WindTurbine:
    def __init__(self, cut_in=3.0, rated=12.0, cut_out=25.0, max_power=2000):
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

def prepare_data(raw_data):
    """Process raw API data into clean DataFrame"""
    hours = len(raw_data['hourly']['time'])
    df = pd.DataFrame({
        "Time": pd.to_datetime(raw_data['hourly']['time']),
        "Wind Speed (m/s)": raw_data['hourly']['wind_speed_10m'],
        "Wind Direction": raw_data['hourly']['wind_direction_10m'],
        "Temperature (°C)": raw_data['hourly']['temperature_2m'],
        "Humidity (%)": raw_data['hourly']['relative_humidity_2m'],
        "Pressure (hPa)": raw_data['hourly']['surface_pressure']
    })
    
    # Add air density
    df['Air Density (kg/m³)'] = calculate_air_density(
        df['Temperature (°C)'], 
        df['Humidity (%)'], 
        df['Pressure (hPa)']
    )
    
    # Mark historical vs forecast data
    now = datetime.now()
    df['Type'] = ['Historical' if t < now else 'Forecast' for t in df['Time']]
    
    return df

def calculate_air_density(temp, humidity, pressure):
    """Calculate air density from weather parameters"""
    R_d = 287.05  # Gas constant for dry air (J/kg·K)
    R_v = 461.495  # Gas constant for water vapor (J/kg·K)
    e_s = 0.61121 * np.exp((18.678 - temp/234.5) * (temp/(257.14 + temp)))
    e = (humidity / 100) * e_s
    rho = (pressure * 100) / (R_d * (temp + 273.15)) * (1 - (0.378 * e) / (pressure * 100))
    return rho

def train_wind_model(df):
    """Train wind speed prediction model"""
    # Feature engineering
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_year'] = df['Time'].dt.dayofyear
    
    # Lag features
    for i in range(1, 4):
        df[f'wind_speed_lag{i}'] = df['Wind Speed (m/s)'].shift(i)
    
    df = df.dropna()
    
    # Features and target
    features = ['hour_sin', 'hour_cos', 'day_of_year', 'Temperature (°C)',
               'Humidity (%)', 'Pressure (hPa)', 'wind_speed_lag1',
               'wind_speed_lag2', 'wind_speed_lag3']
    X = df[features]
    y = df['Wind Speed (m/s)']
    
    # Train/test split (last 20% for testing)
    test_size = int(len(df) * 0.2)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, features, r2, mae

def predict_future(model, features, last_data, hours=48):
    """Predict future wind speeds"""
    predictions = []
    times = []
    current_data = last_data.copy()
    
    for i in range(1, hours+1):
        # Create features for next time step
        next_time = current_data['Time'] + timedelta(hours=1)
        
        # Update lags (shift previous predictions)
        current_data['wind_speed_lag3'] = current_data['wind_speed_lag2']
        current_data['wind_speed_lag2'] = current_data['wind_speed_lag1']
        
        # Prepare feature vector
        features_dict = {
            'hour_sin': np.sin(2 * np.pi * next_time.hour/24),
            'hour_cos': np.cos(2 * np.pi * next_time.hour/24),
            'day_of_year': next_time.timetuple().tm_yday,
            'Temperature (°C)': current_data['Temperature (°C)'],
            'Humidity (%)': current_data['Humidity (%)'],
            'Pressure (hPa)': current_data['Pressure (hPa)'],
            'wind_speed_lag1': current_data['wind_speed_lag1'],
            'wind_speed_lag2': current_data['wind_speed_lag2'],
            'wind_speed_lag3': current_data['wind_speed_lag3']
        }
        
        # Make prediction
        X_pred = pd.DataFrame([features_dict])[features]
        pred_wind = model.predict(X_pred)[0]
        
        # Store results
        predictions.append(pred_wind)
        times.append(next_time)
        
        # Update for next iteration
        current_data['Time'] = next_time
        current_data['wind_speed_lag1'] = pred_wind
    
    return times, predictions

def main():
    st.title("🌬️ Wind Energy Forecast Dashboard")
    st.markdown("""
    **48-hour wind speed and energy generation forecast**  
    Using historical weather patterns to predict future wind conditions
    """)
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("📍 Location", "Chicago, US")
    with col2:
        turbine_power = st.slider("🌀 Turbine Rated Power (kW)", 500, 5000, 2000, step=100)
    
    if st.button("🚀 Generate Forecast"):
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
            df = prepare_data(data)
            turbine = WindTurbine(max_power=turbine_power)
            
            # Train prediction model
            model, features, r2, mae = train_wind_model(df[df['Type'] == 'Historical'].copy())
            
            # Make future predictions
            last_historical = df[df['Type'] == 'Historical'].iloc[-1].to_dict()
            future_times, future_wind = predict_future(model, features, last_historical)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Time': future_times,
                'Wind Speed (m/s)': future_wind,
                'Type': 'Prediction'
            })
            
            # Combine historical and forecast data
            combined_df = pd.concat([
                df[['Time', 'Wind Speed (m/s)', 'Type']],
                forecast_df
            ])
            
            # Calculate power output
            combined_df['Power Output (kW)'] = turbine.power_output(combined_df['Wind Speed (m/s)'])
            
            # Visualization
            st.success(f"✅ Forecast generated for {display_name}")
            
            # Key metrics
            now = datetime.now()
            hist_data = combined_df[combined_df['Time'] < now]
            future_data = combined_df[combined_df['Time'] >= now]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Current Wind Speed", 
                        f"{hist_data['Wind Speed (m/s)'].iloc[-1]:.1f} m/s")
            col2.metric("📈 Predicted Avg Wind Speed (next 48h)", 
                       f"{future_data['Wind Speed (m/s)'].mean():.1f} m/s",
                       f"{((future_data['Wind Speed (m/s)'].mean() - hist_data['Wind Speed (m/s)'].mean())/hist_data['Wind Speed (m/s)'].mean())*100:.1f}%")
            col3.metric("⚡ Predicted Energy (next 48h)", 
                       f"{future_data['Power Output (kW)'].sum()/1000:.1f} MWh")
            
            # Main chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=hist_data['Time'],
                y=hist_data['Wind Speed (m/s)'],
                name='Historical Data',
                line=dict(color='#636EFA', width=2),
                mode='lines'
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=future_data['Time'],
                y=future_data['Wind Speed (m/s)'],
                name='Forecast',
                line=dict(color='#FFA15A', width=3),
                mode='lines'
            ))
            
            # Prediction confidence band
            fig.add_trace(go.Scatter(
                x=future_data['Time'],
                y=future_data['Wind Speed (m/s)'] * 1.1,
                line=dict(width=0),
                showlegend=False,
                mode='lines'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_data['Time'],
                y=future_data['Wind Speed (m/s)'] * 0.9,
                fill='tonexty',
                fillcolor='rgba(255,161,90,0.2)',
                line=dict(width=0),
                name='Confidence Band',
                mode='lines'
            ))
            
            # Now marker
            fig.add_vline(x=now, line_dash="dash", line_color="white",
                         annotation_text="Now", annotation_position="top left")
            
            fig.update_layout(
                title=f"Wind Speed Forecast for {display_name}",
                xaxis_title="Time",
                yaxis_title="Wind Speed (m/s)",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Power output chart
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=combined_df['Time'],
                y=combined_df['Power Output (kW)'],
                name='Power Output',
                line=dict(color='#00CC96', width=2),
                mode='lines'
            ))
            
            fig2.add_vline(x=now, line_dash="dash", line_color="white")
            
            fig2.update_layout(
                title="Turbine Power Output Forecast",
                xaxis_title="Time",
                yaxis_title="Power Output (kW)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Model performance
            with st.expander("🔍 Model Performance Details"):
                st.markdown(f"""
                **Prediction Model Accuracy**:
                - R² Score: {r2:.2f} (1.0 is perfect)
                - Mean Absolute Error: {mae:.2f} m/s
                
                **Forecast Notes**:
                - Historical data shows actual measured wind speeds
                - Forecast uses machine learning trained on past patterns
                - Confidence band shows ±10% prediction uncertainty
                """)
                
                # Actual vs Predicted on test set
                test_data = df[df['Type'] == 'Historical'].iloc[-int(len(df)*0.2):].copy()
                X_test = test_data[features]
                test_data['Predicted'] = model.predict(X_test)
                
                fig3 = px.scatter(
                    test_data,
                    x='Wind Speed (m/s)',
                    y='Predicted',
                    trendline="ols",
                    title="Model Validation: Predicted vs Actual Wind Speeds",
                    labels={'Wind Speed (m/s)': 'Actual', 'Predicted': 'Model Prediction'},
                    template="plotly_dark"
                )
                
                fig3.add_shape(
                    type="line",
                    x0=0, y0=0,
                    x1=test_data['Wind Speed (m/s)'].max(),
                    y1=test_data['Wind Speed (m/s)'].max(),
                    line=dict(color="red", dash="dash")
                )
                
                st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
