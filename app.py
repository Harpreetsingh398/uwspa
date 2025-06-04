import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
from lightgbm import LGBMRegressor

# Configuration
st.set_page_config(layout="wide", page_title="Advanced Wind Analytics", page_icon="🌪️")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0E1117; color: white;}
    .metric-card {border-radius: 10px; padding: 15px; background-color: #1E1E1E; margin: 5px;}
    .highlight {background-color: #2a3f5f; padding: 10px; border-radius: 5px;}
    .insight-card {border-left: 4px solid #4CAF50; padding-left: 10px;}
    .warning-card {border-left: 4px solid #FFA500; padding-left: 10px;}
    .tab-content {padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# API Functions (same as before)
@st.cache_data(ttl=3600)
def get_coordinates(location):
    # ... (previous implementation)

@st.cache_data(ttl=3600)
def get_weather_data(lat, lon, past_days=3):
    # ... (previous implementation)

# Enhanced Turbine Model
class EnhancedWindTurbine(WindTurbine):
    def power_output(self, wind_speed, air_density=1.225):
        # Adjust power for air density (IEC standard)
        wind_speed = np.array(wind_speed)
        power = np.zeros_like(wind_speed)
        adjusted_speed = wind_speed * ((air_density/1.225)**(1/3))
        mask = (adjusted_speed >= self.cut_in) & (adjusted_speed <= self.rated)
        power[mask] = self.max_power * ((adjusted_speed[mask] - self.cut_in)/(self.rated - self.cut_in))**3
        power[adjusted_speed > self.rated] = self.max_power
        power[adjusted_speed > self.cut_out] = 0
        return power

# Enhanced Prediction Model
def create_advanced_model():
    # Feature transformations
    time_transformer = ColumnTransformer([
        ('cyclic_hour', FunctionTransformer(lambda x: np.stack([np.sin(2*np.pi*x/24), 
                                                              np.cos(2*np.pi*x/24)], axis=1)), ['hour']),
        ('cyclic_month', FunctionTransformer(lambda x: np.stack([np.sin(2*np.pi*x/12), 
                                                               np.cos(2*np.pi*x/12)], axis=1)), ['month'])
    ])
    
    # Model pipeline
    return make_pipeline(
        StandardScaler(),
        time_transformer,
        xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    )

def train_advanced_model(df, target='Wind Speed (m/s)'):
    # Feature engineering
    df['hour'] = df['Time'].dt.hour
    df['month'] = df['Time'].dt.month
    df['day_of_year'] = df['Time'].dt.dayofyear
    
    # Lag features with different time windows
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'lag_{lag}'] = df[target].shift(lag)
        df[f'dir_lag_{lag}'] = df['Wind Direction'].shift(lag)
    
    # Rolling statistics
    df['rolling_6h_mean'] = df[target].rolling(6).mean()
    df['rolling_12h_std'] = df[target].rolling(12).std()
    
    # Weather interactions
    df['temp_humidity'] = df['Temperature (°C)'] * df['Humidity (%)']
    df['pressure_temp'] = df['Pressure (hPa)'] * df['Temperature (°C)']
    
    # Drop NA
    df = df.dropna()
    
    # Features
    features = ['hour', 'month', 'day_of_year',
                'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)',
                'lag_1', 'lag_2', 'lag_3', 'lag_24',
                'dir_lag_1', 'dir_lag_3',
                'rolling_6h_mean', 'rolling_12h_std',
                'temp_humidity', 'pressure_temp']
    
    X = df[features]
    y = df[target]
    
    # Time-based validation
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    models = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = create_advanced_model()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        scores.append(score)
        models.append(model)
    
    # Select best model
    best_model = models[np.argmax(scores)]
    test_score = np.mean(scores)
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return best_model, features, test_score, mae, X_test, y_test, y_pred

def predict_with_uncertainty(model, features, last_data, future_hours):
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(future_hours):
        # Prepare input
        input_data = pd.DataFrame([current_data])[features]
        
        # Predict next step
        pred = model.predict(input_data)[0]
        predictions.append(pred)
        
        # Update lags for next prediction
        current_data['lag_1'] = pred
        for lag in [2, 3, 6, 12, 24, 48]:
            if f'lag_{lag}' in current_data:
                current_data[f'lag_{lag}'] = current_data.get(f'lag_{lag-1}', pred)
        
        # Update time features
        current_time = current_data['Time'] + timedelta(hours=1)
        current_data['Time'] = current_time
        current_data['hour'] = current_time.hour
        current_data['month'] = current_time.month
        current_data['day_of_year'] = current_time.timetuple().tm_yday
    
    # Generate uncertainty using bootstrapped residuals
    residuals = y_test - y_pred
    uncertainty = np.random.choice(residuals, size=(100, future_hours), replace=True)
    predictions = np.array(predictions)
    pred_dist = predictions + uncertainty
    
    lower = np.percentile(pred_dist, 5, axis=0)
    upper = np.percentile(pred_dist, 95, axis=0)
    
    future_times = [last_data['Time'] + timedelta(hours=i) for i in range(1, future_hours+1)]
    return future_times, predictions, lower, upper

def main():
    st.title("🌪️ Advanced Wind Energy Analytics")
    
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("📍 Location", "Hamburg, Germany")
            past_days = st.slider("📅 Training Days", 3, 14, 7)
        with col2:
            turbine_model = st.selectbox("🌀 Turbine Model", list(TURBINES.keys()), index=0)
            future_hours = st.slider("🔮 Forecast Hours", 6, 72, 24, step=6)
    
    if st.button("🚀 Run Advanced Analysis"):
        with st.spinner("Running advanced wind pattern analysis..."):
            # Data loading (same as before)
            lat, lon, error, display_name = get_coordinates(location)
            if error:
                st.error(f"❌ {error}")
                return
            
            data = get_weather_data(lat, lon, past_days)
            if 'error' in data:
                st.error(f"❌ {data['error']}")
                return
            
            # Process data
            df = pd.DataFrame({
                "Time": pd.to_datetime(data['hourly']['time']),
                "Wind Speed (m/s)": data['hourly']['wind_speed_10m'],
                "Wind Direction": data['hourly']['wind_direction_10m'],
                "Temperature (°C)": data['hourly']['temperature_2m'],
                "Humidity (%)": data['hourly']['relative_humidity_2m'],
                "Pressure (hPa)": data['hourly']['surface_pressure']
            })
            
            # Enhanced calculations
            df['Air Density (kg/m³)'] = calculate_air_density(
                df['Temperature (°C)'], df['Humidity (%)'], df['Pressure (hPa)'])
            
            turbine = EnhancedWindTurbine(
                TURBINES[turbine_model].name,
                TURBINES[turbine_model].cut_in,
                TURBINES[turbine_model].rated,
                TURBINES[turbine_model].cut_out,
                TURBINES[turbine_model].max_power,
                TURBINES[turbine_model].rotor_diam
            )
            
            df['Power Output (kW)'] = turbine.power_output(
                df['Wind Speed (m/s)'], df['Air Density (kg/m³)'])
            
            # Time series decomposition
            df.set_index('Time', inplace=True)
            try:
                decomposition = seasonal_decompose(
                    df['Wind Speed (m/s)'].resample('H').mean(),
                    period=24, model='additive')
                seasonal_component = decomposition.seasonal
                trend_component = decomposition.trend
            except:
                seasonal_component = None
                trend_component = None
            
            df.reset_index(inplace=True)
            
            # Train model
            model, features, test_score, mae, X_test, y_test, y_pred = train_advanced_model(df)
            
            # Make predictions
            last_data = df.iloc[-1].to_dict()
            future_times, pred_speeds, lower, upper = predict_with_uncertainty(
                model, features, last_data, future_hours)
            
            pred_df = pd.DataFrame({
                'Time': future_times,
                'Predicted Wind Speed (m/s)': pred_speeds,
                'Lower Bound': lower,
                'Upper Bound': upper
            })
            
            pred_df['Predicted Power (kW)'] = turbine.power_output(
                pred_df['Predicted Wind Speed (m/s)'],
                last_data['Air Density (kg/m³)']  # Using last known air density
            )
            
            # Combine data
            past_df = df[df['Time'] < datetime.now()]
            combined_df = pd.concat([
                past_df[['Time', 'Wind Speed (m/s)', 'Power Output (kW)']],
                pred_df[['Time', 'Predicted Wind Speed (m/s)', 'Predicted Power (kW)']]
            ])
            
            # Dashboard Layout
            tab1, tab2, tab3 = st.tabs([
                "🌪️ Wind Analysis", 
                "🔮 Predictive Insights", 
                "⚡ Energy Forecast"
            ])
            
            with tab1:
                st.subheader("Comprehensive Wind Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Wind Speed Timeline
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=past_df['Time'],
                        y=past_df['Wind Speed (m/s)'],
                        name='Historical',
                        line=dict(color='#636EFA', width=2)
                    ))
                    
                    # Predictions
                    fig.add_trace(go.Scatter(
                        x=pred_df['Time'],
                        y=pred_df['Predicted Wind Speed (m/s)'],
                        name='Predicted',
                        line=dict(color='#FFA15A', width=3)
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=pred_df['Time'],
                        y=pred_df['Upper Bound'],
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_df['Time'],
                        y=pred_df['Lower Bound'],
                        fill='tonexty',
                        fillcolor='rgba(255,161,90,0.2)',
                        line=dict(width=0),
                        name='90% Confidence'
                    ))
                    
                    # Add vertical line for now
                    fig.add_vline(
                        x=datetime.now(),
                        line_dash="dash",
                        line_color="white",
                        annotation_text="Now"
                    )
                    
                    fig.update_layout(
                        title="Wind Speed Timeline with Probabilistic Forecast",
                        xaxis_title="Time",
                        yaxis_title="Wind Speed (m/s)",
                        template="plotly_dark",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights
                    with st.expander("🔍 Key Insights", expanded=True):
                        st.markdown(f"""
                        <div class="insight-card">
                        <h4>📌 Wind Pattern Analysis</h4>
                        <ul>
                            <li>Average wind speed: <b>{past_df['Wind Speed (m/s)'].mean():.1f} m/s</b></li>
                            <li>Max gust: <b>{past_df['Wind Speed (m/s)'].max():.1f} m/s</b></li>
                            <li>Current trend: <b>{'increasing' if pred_speeds[0] > past_df['Wind Speed (m/s)'].iloc[-1] else 'decreasing'}</b></li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if seasonal_component is not None:
                            st.markdown(f"""
                            <div class="insight-card">
                            <h4>⏱️ Diurnal Pattern</h4>
                            <p>Wind shows a <b>{'strong' if seasonal_component.max() > 1.5 else 'moderate'}</b> 
                            daily cycle with peak winds around <b>{seasonal_component.idxmax().hour}:00</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    # Wind Direction Analysis
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=past_df['Wind Speed (m/s)'],
                        theta=past_df['Wind Direction'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=past_df['Temperature (°C)'],
                            colorscale='thermal',
                            showscale=True,
                            colorbar=dict(title='Temp (°C)')
                        )
                    ))
                    fig.update_layout(
                        title="Wind Direction & Temperature Analysis",
                        polar=dict(
                            radialaxis=dict(visible=True),
                            angularaxis=dict(direction="clockwise")
                        ),
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Power Curve
                    fig = go.Figure()
                    wind_range = np.linspace(0, turbine.cut_out*1.2, 100)
                    fig.add_trace(go.Scatter(
                        x=wind_range,
                        y=turbine.power_output(wind_range),
                        name='Standard Power Curve'
                    ))
                    fig.add_trace(go.Scatter(
                        x=wind_range,
                        y=turbine.power_output(wind_range, 1.15),
                        name='High Density (1.15 kg/m³)',
                        line=dict(dash='dot')
                    ))
                    fig.update_layout(
                        title=f"{turbine_model} Power Curve",
                        xaxis_title="Wind Speed (m/s)",
                        yaxis_title="Power Output (kW)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Advanced Predictive Insights")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Model Performance
                    st.metric("Model Accuracy (R²)", f"{test_score:.2%}")
                    st.metric("Mean Absolute Error", f"{mae:.2f} m/s")
                    
                    # Feature Importance
                    importance = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.steps[-1][1].feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig = px.bar(importance, x='Importance', y='Feature',
                                title="Top Predictive Features",
                                template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction Diagnostics
                    fig = px.scatter(
                        x=y_test,
                        y=y_pred,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title="Model Validation",
                        trendline="ols",
                        template="plotly_dark"
                    )
                    fig.add_shape(
                        type="line",
                        x0=0, y0=0,
                        x1=max(y_test.max(), y_pred.max()),
                        y1=max(y_test.max(), y_pred.max()),
                        line=dict(color="Red", width=2, dash="dash")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Forecast Breakdown
                    st.markdown("### 📅 Forecast Breakdown")
                    
                    # Next 24 hours
                    next_24h = pred_df[pred_df['Time'] <= (datetime.now() + timedelta(hours=24))]
                    avg_speed = next_24h['Predicted Wind Speed (m/s)'].mean()
                    max_speed = next_24h['Predicted Wind Speed (m/s)'].max()
                    
                    col1, col2 = st.columns(2)
                    col1.metric("24h Avg Speed", f"{avg_speed:.1f} m/s")
                    col2.metric("24h Peak Gust", f"{max_speed:.1f} m/s")
                    
                    # Power generation potential
                    capacity_factor = next_24h['Predicted Power (kW)'].mean() / turbine.max_power * 100
                    st.metric("Expected Capacity Factor", f"{capacity_factor:.1f}%")
                    
                    # Optimal generation hours
                    optimal_hours = next_24h[next_24h['Predicted Power (kW)'] > turbine.max_power * 0.8]
                    st.markdown(f"""
                    <div class="highlight">
                    <h4>🏆 Peak Generation Window</h4>
                    <p>Best production hours: <b>{len(optimal_hours)} hours</b> at >80% capacity</p>
                    {f"<p>From {optimal_hours['Time'].iloc[0].strftime('%H:%M')} to {optimal_hours['Time'].iloc[-1].strftime('%H:%M')}</p>" if len(optimal_hours) > 0 else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Warning system
                    if max_speed > turbine.cut_out * 0.9:
                        st.markdown(f"""
                        <div class="warning-card">
                        <h4>⚠️ Turbine Shutdown Risk</h4>
                        <p>Predicted wind speeds may approach cut-out speed ({turbine.cut_out}m/s)</p>
                        <p>Consider preemptive measures if gusts exceed {turbine.cut_out * 0.95:.1f}m/s</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
                st.subheader("Energy Production Forecast")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Cumulative Energy
                    combined_df['Cumulative Energy (kWh)'] = np.concatenate([
                        past_df['Power Output (kW)'].cumsum().values,
                        (past_df['Power Output (kW)'].sum() + 
                         pred_df['Predicted Power (kW)'].cumsum()).values
                    ])
                    
                    fig = px.area(
                        combined_df,
                        x="Time",
                        y="Cumulative Energy (kWh)",
                        title="Projected Energy Output",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Energy Summary
                    total_energy = combined_df['Cumulative Energy (kWh)'].iloc[-1]
                    daily_energy = total_energy / (len(pred_df)/24)
                    st.metric("Total Projected Energy", f"{total_energy/1000:.1f} MWh")
                    st.metric("Daily Energy Potential", f"{daily_energy/1000:.1f} MWh/day")
                
                with col2:
                    # Power Duration Curve
                    power_sorted = np.sort(combined_df['Predicted Power (kW)'].dropna())[::-1]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(power_sorted))/len(power_sorted)*100,
                        y=power_sorted,
                        mode='lines'
                    ))
                    fig.update_layout(
                        title="Power Duration Curve (Predicted)",
                        xaxis_title="Percentage of Time",
                        yaxis_title="Power Output (kW)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Energy Value Insights
                    with st.expander("💡 Energy Value Analysis", expanded=True):
                        st.markdown(f"""
                        <div class="insight-card">
                        <h4>💰 Revenue Potential</h4>
                        <p>Assuming $0.05/kWh market price:</p>
                        <ul>
                            <li>24h revenue: <b>${daily_energy*0.05:,.0f}</b></li>
                            <li>Monthly potential: <b>${daily_energy*0.05*30:,.0f}</b></li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
