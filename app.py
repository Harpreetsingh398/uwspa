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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
    .tab-content {padding: 15px; border-radius: 10px; background-color: #1E1E1E;}
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
def get_weather_data(lat, lon, past_days=2):
    """Get historical weather data for model training"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure&past_days={past_days}"
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

# Enhanced Wind Speed Prediction Model
def train_wind_speed_model(df):
    # Feature engineering
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['day_of_year'] = df['Time'].dt.dayofyear
    df['month'] = df['Time'].dt.month
    
    # Rolling features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'wind_speed_lag_{lag}'] = df['Wind Speed (m/s)'].shift(lag)
        df[f'wind_dir_lag_{lag}'] = df['Wind Direction'].shift(lag)
    
    # Weather interaction features
    df['temp_humidity'] = df['Temperature (°C)'] * df['Humidity (%)']
    df['pressure_temp'] = df['Pressure (hPa)'] * df['Temperature (°C)']
    
    # Drop NA values from lag features
    df = df.dropna()
    
    # Feature selection
    features = ['hour_sin', 'hour_cos', 'day_of_week', 'month',
                'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)',
                'wind_speed_lag_1', 'wind_speed_lag_2', 'wind_speed_lag_3',
                'wind_speed_lag_24', 'wind_dir_lag_1', 'temp_humidity']
    
    X = df[features]
    y = df['Wind Speed (m/s)']
    
    # Train/test split with time-based validation
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Enhanced model with feature scaling
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            n_jobs=-1,
            random_state=42
        )
    )
    
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    
    return model, features, train_score, test_score, mae

def predict_future_wind(model, features, last_data_point, future_hours):
    future_times = [last_data_point['Time'] + timedelta(hours=i) for i in range(1, future_hours+1)]
    
    pred_data = []
    for i, time in enumerate(future_times):
        # Use last known values for lag features
        row = {
            'Time': time,
            'hour_sin': np.sin(2 * np.pi * time.hour/24),
            'hour_cos': np.cos(2 * np.pi * time.hour/24),
            'day_of_week': time.weekday(),
            'month': time.month,
            'Temperature (°C)': last_data_point['Temperature (°C)'],
            'Humidity (%)': last_data_point['Humidity (%)'],
            'Pressure (hPa)': last_data_point['Pressure (hPa)'],
            'wind_speed_lag_1': last_data_point['Wind Speed (m/s)'],
            'wind_speed_lag_2': last_data_point.get('wind_speed_lag_1', last_data_point['Wind Speed (m/s)']),
            'wind_speed_lag_3': last_data_point.get('wind_speed_lag_2', last_data_point['Wind Speed (m/s)']),
            'wind_speed_lag_24': last_data_point.get('wind_speed_lag_24', last_data_point['Wind Speed (m/s)']),
            'wind_dir_lag_1': last_data_point['Wind Direction'],
            'temp_humidity': last_data_point['Temperature (°C)'] * last_data_point['Humidity (%)']
        }
        pred_data.append(row)
    
    pred_df = pd.DataFrame(pred_data)
    
    # Make predictions with confidence intervals using bootstrap
    preds = []
    for _ in range(100):  # Bootstrap samples
        pred = model.predict(pred_df[features])
        preds.append(pred)
    
    preds = np.array(preds)
    mean_pred = np.mean(preds, axis=0)
    lower_bound = np.percentile(preds, 5, axis=0)
    upper_bound = np.percentile(preds, 95, axis=0)
    
    return future_times, mean_pred, lower_bound, upper_bound

def main():
    st.title("🌬️ Wind Energy Analytics Dashboard")
    
    with st.expander("⚙️ Configuration Panel", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("📍 Location", "Chennai, India")
            past_days = st.slider("📅 Past Days for Training", 1, 7, 3)
        with col2:
            turbine_model = st.selectbox("🌀 Turbine Model", list(TURBINES.keys()), index=0)
            future_hours = st.slider("🔮 Hours to Forecast", 6, 48, 24, step=6)
    
    if st.button("🚀 Analyze Wind Data"):
        with st.spinner("Fetching data and training prediction model..."):
            # Get coordinates
            lat, lon, error, display_name = get_coordinates(location)
            if error:
                st.error(f"❌ {error}")
                return
                
            st.success(f"🔍 Location found: {display_name} (Latitude: {lat:.4f}, Longitude: {lon:.4f})")
            
            # Get historical weather data
            data = get_weather_data(lat, lon, past_days=past_days)
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
            
            # Calculate air density and power output
            df['Air Density (kg/m³)'] = calculate_air_density(
                df['Temperature (°C)'], 
                df['Humidity (%)'], 
                df['Pressure (hPa)']
            )
            
            turbine = TURBINES[turbine_model]
            df['Power Output (kW)'] = turbine.power_output(df['Wind Speed (m/s)'])
            
            # Split into training period and forecast period
            now = datetime.now()
            past_df = df[df['Time'] < now].copy()
            current_time = past_df['Time'].max() if not past_df.empty else now
            
            # Train model on past data
            if len(past_df) < 24:  # Minimum data requirement
                st.error("❌ Insufficient historical data for accurate predictions (need at least 24 hours)")
                return
                
            model, features, train_score, test_score, mae = train_wind_speed_model(past_df)
            
            # Make predictions
            last_point = past_df.iloc[-1].to_dict()
            pred_times, pred_speeds, lower_bounds, upper_bounds = predict_future_wind(
                model, features, last_point, future_hours
            )
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'Time': pred_times,
                'Predicted Wind Speed (m/s)': pred_speeds,
                'Lower Bound': lower_bounds,
                'Upper Bound': upper_bounds
            })
            
            # Calculate power output for predictions
            pred_df['Predicted Power (kW)'] = turbine.power_output(pred_df['Predicted Wind Speed (m/s)'])
            
            # Combine past and future for visualization
            combined_df = pd.concat([
                past_df[['Time', 'Wind Speed (m/s)', 'Power Output (kW)']],
                pred_df[['Time', 'Predicted Wind Speed (m/s)', 'Predicted Power (kW)', 'Lower Bound', 'Upper Bound']]
            ])
            
            # Weibull distribution fit
            wind_speeds = past_df['Wind Speed (m/s)']
            hist, bin_edges = np.histogram(wind_speeds, bins=20, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            try:
                params, _ = curve_fit(weibull, bin_centers, hist, p0=[2, 6])
                k, A = params
            except:
                k, A = 2, 6  # Default values if fit fails
            
            # Dashboard Tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Wind Analysis", 
                "🌀 Turbine Performance", 
                "⚡ Energy Forecast", 
                "🔮 Advanced Prediction"
            ])
            
            with tab1:
                st.subheader("🌪️ Wind Characteristics Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Wind Speed Timeline
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=past_df['Time'],
                        y=past_df['Wind Speed (m/s)'],
                        name='Historical Data',
                        line=dict(color='#636EFA', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_df['Time'],
                        y=pred_df['Predicted Wind Speed (m/s)'],
                        name='Predicted Wind Speed',
                        line=dict(color='#FFA15A', width=3)
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
                        fillcolor='rgba(255,161,90,0.2)',
                        line=dict(width=0),
                        name='Confidence Interval',
                        mode='lines'
                    ))
                    fig.update_layout(
                        title="Wind Speed Timeline with Predictions",
                        xaxis_title="Time",
                        yaxis_title="Wind Speed (m/s)",
                        template="plotly_dark",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Wind Direction Analysis
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=past_df['Wind Speed (m/s)'],
                        theta=past_df['Wind Direction'],
                        mode='markers',
                        name='Historical',
                        marker=dict(
                            size=6,
                            color=past_df['Wind Speed (m/s)'],
                            colorscale='Viridis',
                            showscale=True
                        )
                    ))
                    fig.update_layout(
                        title="Wind Direction Analysis (Historical Data)",
                        polar=dict(radialaxis=dict(visible=True)),
                        template="plotly_dark",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Wind Speed Distribution
                    fig = px.histogram(past_df, x="Wind Speed (m/s)", nbins=20,
                                     title="Wind Speed Distribution (Historical Data)",
                                     marginal="rug",
                                     template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Weibull Distribution
                    x = np.linspace(0, past_df['Wind Speed (m/s)'].max()*1.2, 100)
                    y = weibull(x, k, A)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, name="Weibull Fit"))
                    fig.add_trace(go.Histogram(x=past_df['Wind Speed (m/s)'], histnorm='probability density', 
                                            name="Actual Data", opacity=0.5))
                    fig.update_layout(
                        title=f"Weibull Distribution (k={k:.2f}, A={A:.2f})",
                        xaxis_title="Wind Speed (m/s)",
                        yaxis_title="Probability Density",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("🌀 Turbine Performance Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Power Output Timeline
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=past_df['Time'],
                        y=past_df['Power Output (kW)'],
                        name='Historical Power',
                        line=dict(color='#00CC96', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_df['Time'],
                        y=pred_df['Predicted Power (kW)'],
                        name='Predicted Power',
                        line=dict(color='#EF553B', width=3)
                    ))
                    fig.update_layout(
                        title="Power Output Timeline",
                        xaxis_title="Time",
                        yaxis_title="Power Output (kW)",
                        template="plotly_dark",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Power Curve
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
            
            with tab3:
                st.subheader("⚡ Energy Production Forecast")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Cumulative Energy
                    combined_df['Cumulative Energy (kWh)'] = np.concatenate([
                        past_df['Power Output (kW)'].cumsum().values,
                        (past_df['Power Output (kW)'].sum() + pred_df['Predicted Power (kW)'].cumsum()).values
                    ])
                    
                    fig = px.area(combined_df, x="Time", y="Cumulative Energy (kWh)", 
                                 title="Cumulative Energy Production",
                                 template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Daily Pattern
                    combined_df['Hour'] = combined_df['Time'].dt.hour
                    hourly_avg = combined_df.groupby('Hour').agg({
                        'Predicted Wind Speed (m/s)': 'mean',
                        'Predicted Power (kW)': 'mean'
                    }).reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=hourly_avg['Hour'],
                        y=hourly_avg['Predicted Power (kW)'],
                        name='Average Power'
                    ))
                    fig.add_trace(go.Scatter(
                        x=hourly_avg['Hour'],
                        y=hourly_avg['Predicted Wind Speed (m/s)'],
                        name='Wind Speed',
                        yaxis="y2"
                    ))
                    fig.update_layout(
                        title="Diurnal Pattern of Wind and Power",
                        xaxis_title="Hour of Day",
                        yaxis_title="Power Output (kW)",
                        yaxis2=dict(title="Wind Speed (m/s)", overlaying="y", side="right"),
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("🔮 Advanced Prediction Analytics")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Model Performance
                    st.metric("Model Training R²", f"{train_score:.2%}")
                    st.metric("Model Test R²", f"{test_score:.2%}")
                    st.metric("Mean Absolute Error", f"{mae:.2f} m/s")
                
                with col2:
                    # Feature Importance
                    feature_imp = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.steps[1][1].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_imp.head(10), 
                                x='Importance', y='Feature',
                                title="Top 10 Important Features",
                                template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Prediction Diagnostics
                st.subheader("Prediction Diagnostics")
                
                # Actual vs Predicted on training data
                train_pred = model.predict(past_df[features])
                fig = px.scatter(
                    x=past_df['Wind Speed (m/s)'],
                    y=train_pred,
                    labels={'x': 'Actual Wind Speed', 'y': 'Predicted Wind Speed'},
                    title="Model Fit on Training Data",
                    trendline="ols",
                    template="plotly_dark"
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=0,
                    x1=max(past_df['Wind Speed (m/s)']),
                    y1=max(past_df['Wind Speed (m/s)']),
                    line=dict(color="Red", width=2, dash="dash")
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
