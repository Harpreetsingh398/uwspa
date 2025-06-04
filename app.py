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
def get_weather_data(lat, lon, past_days=3):
    """Get weather data with validation"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days={past_days}"
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
    # Create future timestamps starting from the last data point
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
        col1, col2, col3 = st.columns(3)
        with col1:
            location = st.text_input("📍 Location", "Chennai, India", 
                                   help="Enter a valid city name and country (e.g., 'New York, US')")
            past_days = st.slider("📅 Days of Historical Data", 1, 7, 3)
            
        with col2:
            turbine_model = st.selectbox("🌀 Turbine Model", list(TURBINES.keys()), index=0)
            if turbine_model == "Custom Turbine":
                st.number_input("Cut-in Speed (m/s)", min_value=1.0, max_value=10.0, value=3.0)
                st.number_input("Rated Speed (m/s)", min_value=5.0, max_value=20.0, value=12.0)
                st.number_input("Cut-out Speed (m/s)", min_value=15.0, max_value=30.0, value=25.0)
                st.number_input("Rated Power (kW)", min_value=100, max_value=10000, value=2000)
        
        with col3:
            analysis_type = st.selectbox("📊 Analysis Type", 
                                      ["Basic Forecast", "Technical Analysis", "Financial Evaluation"])
            show_raw_data = st.checkbox("📋 Show Raw Data CSV", False)
            future_hours = st.slider("Hours to Forecast Ahead", 6, 48, 24, step=6,
                                   help="Number of hours to predict wind speed into the future")
    
    if st.button("🚀 Analyze Wind Data"):
        with st.spinner("Fetching wind data and performing analysis..."):
            # Data Acquisition with strict validation
            lat, lon, error, display_name = get_coordinates(location)
            
            if error:
                st.error(f"❌ {error}")
                return
                
            st.success(f"🔍 Location found: {display_name} (Latitude: {lat:.4f}, Longitude: {lon:.4f})")
            
            data = get_weather_data(lat, lon, past_days)
            if 'error' in data:
                st.error(f"❌ Weather API Error: {data['error']}")
                return
            
            # Data Processing
            hours = (past_days + 1) * 24  # Past days + today
            times = pd.to_datetime(data['hourly']['time'])
            df = pd.DataFrame({
                "Time": times,
                "Wind Speed (m/s)": data['hourly']['wind_speed_10m'],
                "Wind Direction": data['hourly']['wind_direction_10m'],
                "Temperature (°C)": data['hourly']['temperature_2m'],
                "Humidity (%)": data['hourly']['relative_humidity_2m'],
                "Pressure (hPa)": data['hourly']['surface_pressure']
            })
            
            # Filter to only keep data up to current hour (for realistic historical data)
            now = datetime.now()
            df = df[df['Time'] <= now]
            
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
            
            # Weibull Distribution Fit
            wind_speeds = df['Wind Speed (m/s)']
            hist, bin_edges = np.histogram(wind_speeds, bins=20, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            try:
                params, _ = curve_fit(weibull, bin_centers, hist, p0=[2, 6])
                k, A = params
            except:
                k, A = 2, 6  # Default values if fit fails
            
            # Train wind speed prediction model with enhanced features
            model, features, test_accuracy = train_wind_speed_model(df.copy())
            
            # Get the last data point for prediction
            last_data_point = df.iloc[-1].to_dict()
            
            # Predict future wind speeds
            future_times, future_wind = predict_future_wind(model, features, last_data_point, future_hours)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Time': future_times,
                'Predicted Wind Speed (m/s)': future_wind,
                'Lower Bound': future_wind * 0.95,  # 5% lower
                'Upper Bound': future_wind * 1.05   # 5% higher
            })
            
            # Combine historical and predicted data
            combined_df = pd.concat([
                df[['Time', 'Wind Speed (m/s)', 'Wind Direction', 'Temperature (°C)', 
                    'Humidity (%)', 'Pressure (hPa)', 'Air Density (kg/m³)', 
                    'Power Output (kW)', 'Energy Output (kWh)']],
                pred_df
            ], ignore_index=True)
            
            # Mark data types (historical vs predicted)
            combined_df['Data Type'] = 'Historical'
            combined_df.loc[combined_df['Time'] > now, 'Data Type'] = 'Predicted'
            
            # Dashboard Layout
            st.success(f"✅ Analysis completed for {display_name}")
            
            # Key Metrics
            st.subheader("📊 Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡️ Average Wind Speed", f"{df['Wind Speed (m/s)'].mean():.2f} m/s")
            col2.metric("💨 Max Wind Speed", f"{df['Wind Speed (m/s)'].max():.2f} m/s")
            col3.metric("⚡ Total Energy Output", f"{df['Energy Output (kWh)'].sum()/1000:.2f} MWh")
            col4.metric("🌀 Predominant Direction", f"{df['Wind Direction'].mode()[0]}°")
            
            # Main Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Wind Analysis", "Turbine Performance", "Energy Forecast", "Wind Prediction"])
            
            with tab1:
                st.subheader("🌪️ Wind Characteristics Analysis")
                
                # Wind Speed Timeline
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Historical']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Historical']['Wind Speed (m/s)'],
                    name='Historical Data',
                    line=dict(color='#1f77b4')
                ))
                
                # Predicted data
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Predicted']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Predicted']['Predicted Wind Speed (m/s)'],
                    name='Predicted Data',
                    line=dict(color='#ff7f0e')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Predicted']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Predicted']['Upper Bound'],
                    line=dict(width=0),
                    showlegend=False,
                    mode='lines'
                ))
                
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Predicted']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Predicted']['Lower Bound'],
                    fill='tonexty',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(width=0),
                    name='Confidence Interval',
                    mode='lines'
                ))
                
                # Add vertical line for current time
                fig.add_vline(x=now, line_dash="dash", line_color="red", 
                              annotation_text="Current Time", annotation_position="top left")
                
                fig.update_layout(
                    title="Wind Speed Timeline (Historical + Predicted)",
                    xaxis_title="Time",
                    yaxis_title="Wind Speed (m/s)",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Other wind analysis charts
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x="Wind Speed (m/s)", nbins=20,
                                     title="Wind Speed Distribution",
                                     marginal="rug",
                                     template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    x = np.linspace(0, df['Wind Speed (m/s)'].max()*1.2, 100)
                    y = weibull(x, k, A)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, name="Weibull Fit"))
                    fig.add_trace(go.Histogram(x=df['Wind Speed (m/s)'], histnorm='probability density', 
                                            name="Actual Data", opacity=0.5))
                    fig.update_layout(
                        title=f"Weibull Distribution (k={k:.2f}, A={A:.2f})",
                        xaxis_title="Wind Speed (m/s)",
                        yaxis_title="Probability Density",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(df, x="Temperature (°C)", y="Wind Speed (m/s)", 
                                   color="Humidity (%)",
                                   title="Weather Impact Analysis",
                                   trendline="lowess",
                                   template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.density_heatmap(df, x="Time", y="Wind Speed (m/s)", 
                                           title="Wind Speed Patterns",
                                           nbinsx=24*past_days, nbinsy=20,
                                           template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("🌀 Turbine Performance Analysis")
                
                # Power Output Timeline
                fig = go.Figure()
                
                # Historical power
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Historical']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Historical']['Power Output (kW)'],
                    name='Historical Power',
                    line=dict(color='#2ca02c')
                ))
                
                # Predicted power (using predicted wind speeds)
                pred_power = turbine.power_output(combined_df[combined_df['Data Type'] == 'Predicted']['Predicted Wind Speed (m/s)'])
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Predicted']['Time'],
                    y=pred_power,
                    name='Predicted Power',
                    line=dict(color='#d62728')
                ))
                
                # Add vertical line for current time
                fig.add_vline(x=now, line_dash="dash", line_color="red", 
                              annotation_text="Current Time", annotation_position="top left")
                
                fig.update_layout(
                    title="Turbine Power Output Timeline",
                    xaxis_title="Time",
                    yaxis_title="Power Output (kW)",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Other turbine performance charts
                col1, col2 = st.columns(2)
                with col1:
                    wind_range = np.linspace(0, turbine.cut_out*1.2, 100)
                    power_curve = turbine.power_output(wind_range)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=wind_range, y=power_curve, name="Power Curve"))
                    fig.add_vline(x=turbine.cut_in, line_dash="dash", annotation_text=f"Cut-in: {turbine.cut_in}m/s")
                    fig.add_vline(x=turbine.rated, line_dash="dash", annotation_text=f"Rated: {turbine.rated}m/s")
                    fig.add_vline(x=turbine.cut_out, line_dash="dash", annotation_text=f"Cut-out: {turbine.cut_out}m/s")
                    fig.update_layout(
                        title=f"{turbine_model} Power Curve",
                        xaxis_title="Wind Speed (m/s)",
                        yaxis_title="Power Output (kW)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(df, x="Wind Speed (m/s)", y="Power Output (kW)", 
                                    color="Air Density (kg/m³)",
                                    title="Power-Wind Relationship",
                                    trendline="lowess",
                                    template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("⚡ Energy Production Forecast")
                
                # Energy Output Timeline
                fig = go.Figure()
                
                # Historical energy
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Historical']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Historical']['Energy Output (kWh)'].cumsum(),
                    name='Historical Energy',
                    line=dict(color='#9467bd')
                ))
                
                # Predicted energy (using predicted power)
                pred_energy = pred_power.cumsum() + df['Energy Output (kWh)'].sum()
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Predicted']['Time'],
                    y=pred_energy,
                    name='Predicted Energy',
                    line=dict(color='#e377c2')
                ))
                
                # Add vertical line for current time
                fig.add_vline(x=now, line_dash="dash", line_color="red", 
                              annotation_text="Current Time", annotation_position="top left")
                
                fig.update_layout(
                    title="Cumulative Energy Production",
                    xaxis_title="Time",
                    yaxis_title="Cumulative Energy (kWh)",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Other energy charts
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.box(df, x=df['Time'].dt.day_name(), y="Energy Output (kWh)", 
                               title="Daily Energy Variability",
                               color=df['Time'].dt.day_name(),
                               template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    capacity_factor = (df['Energy Output (kWh)'].sum() / (turbine.max_power * len(df))) * 100
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=capacity_factor,
                        title="Capacity Factor",
                        gauge={'axis': {'range': [0, 100]}},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("🔮 Advanced Wind Speed Prediction")
                
                # Model performance metrics
                st.markdown(f"""
                ### Prediction Model Performance
                - **Model Type**: Random Forest Regressor (200 trees)
                - **Test Accuracy (R²)**: {test_accuracy:.2%}
                - **Mean Absolute Error**: {mean_absolute_error(df['Wind Speed (m/s)'], model.predict(df[features])):.2f} m/s
                """)
                
                # Prediction visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Historical']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Historical']['Wind Speed (m/s)'],
                    name='Historical Data',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Predicted data
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Predicted']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Predicted']['Predicted Wind Speed (m/s)'],
                    name='Prediction',
                    line=dict(color='#ff7f0e', width=3)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Predicted']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Predicted']['Upper Bound'],
                    line=dict(width=0),
                    showlegend=False,
                    mode='lines'
                ))
                
                fig.add_trace(go.Scatter(
                    x=combined_df[combined_df['Data Type'] == 'Predicted']['Time'],
                    y=combined_df[combined_df['Data Type'] == 'Predicted']['Lower Bound'],
                    fill='tonexty',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(width=0),
                    name='Confidence Interval',
                    mode='lines'
                ))
                
                # Add vertical line for current time
                fig.add_vline(x=now, line_dash="dash", line_color="red", 
                              annotation_text="Current Time", annotation_position="top left")
                
                fig.update_layout(
                    title=f"Wind Speed Forecast - Next {future_hours} Hours",
                    xaxis_title="Time",
                    yaxis_title="Wind Speed (m/s)",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction metrics
                avg_wind = combined_df[combined_df['Data Type'] == 'Predicted']['Predicted Wind Speed (m/s)'].mean()
                max_wind = combined_df[combined_df['Data Type'] == 'Predicted']['Predicted Wind Speed (m/s)'].max()
                
                col1, col2 = st.columns(2)
                col1.metric("Average Predicted Wind Speed", f"{avg_wind:.2f} m/s")
                col2.metric("Maximum Predicted Wind Speed", f"{max_wind:.2f} m/s")

if __name__ == "__main__":
    main()
