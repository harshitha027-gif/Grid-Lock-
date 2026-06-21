import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.risk_engine import predict_event, calculate_risk_score, compute_factor_breakdown
from src.resource_recommender import recommend_resources
from src.diversion_engine import DiversionEngine
from src.config import (
    RISK_WEIGHTS, BENGALURU_LAT_RANGE, BENGALURU_LON_RANGE,
    DENSITY_RADIUS_KM, EARTH_RADIUS_KM,
)

# Set Page Config
st.set_page_config(
    page_title="Astram - AI Traffic Congestion Management Platform",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Slate/Dark Tech Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    .title-gradient {
        background: linear-gradient(90deg, #3b82f6, #10b981, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* KPI card styling */
    .kpi-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        text-align: center;
    }
    .kpi-val {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .kpi-lbl {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk badge styles */
    .risk-badge {
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        font-size: 1.2rem;
    }
    .risk-low { background-color: #065f46; color: #34d399; border: 1px solid #059669; }
    .risk-medium { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .risk-high { background-color: #7c2d12; color: #fb923c; border: 1px solid #ea580c; }
    .risk-critical { background-color: #991b1b; color: #fca5a5; border: 1px solid #dc2626; }
    
    /* Resource item styling */
    .res-card {
        background-color: #0f172a;
        border-left: 5px solid #3b82f6;
        padding: 10px 15px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 10px;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    processed_path = "data/processed/processed_traffic_events.csv"
    coords_path = "data/processed/historical_coords.csv"
    raw_path = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    
    if os.path.exists(processed_path) and os.path.exists(coords_path):
        df = pd.read_csv(processed_path)
        # Parse datetimes
        df['start_datetime'] = pd.to_datetime(df['start_datetime'])
        coords_df = pd.read_csv(coords_path)
        return df, coords_df, False
    elif os.path.exists(raw_path):
        # Fallback processing if pipeline hasn't been run (Fix 4.1: add spinner)
        with st.spinner("⏳ Processed dataset not found. Running on-the-fly preprocessing — this may take 30-60 seconds..."):
            df = preprocess_data(raw_path)
            df = engineer_features(df, is_training=True)
            df['risk_score'], df['risk_level'] = calculate_risk_score(df)
            coords_df = df[['latitude', 'longitude', 'junction', 'zone']].drop_duplicates()
        return df, coords_df, True
    else:
        st.error("❌ Dataset not found! Please ensure 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv' is in the workspace.")
        return None, None, False

df_all, coords_df, is_fallback = load_data()

# Initialize Diversion Engine
@st.cache_resource
def get_diversion_engine(coords_data):
    if coords_data is not None:
        return DiversionEngine(coords_data)
    return None

# Fix 4.2: use deduplicated coords_df, not full df_all (avoids duplicate BallTree entries)
diversion_engine = get_diversion_engine(coords_df)

# Sidebar Control Center
st.sidebar.markdown("<h2 style='text-align: center; color: #3b82f6;'>🚦 ASTRAM</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>AI-Driven Event Traffic Intelligence</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Fix 4.4: initialize filtered_df before the conditional to prevent NameError
filtered_df = pd.DataFrame()

if df_all is not None:
    # Sidebar Filters
    st.sidebar.subheader("Filter Incidents")
    
    event_types = ['All'] + list(df_all['event_type'].unique())
    selected_type = st.sidebar.selectbox("Event Type", event_types)
    
    event_causes = ['All'] + list(df_all['event_cause'].unique())
    selected_cause = st.sidebar.selectbox("Event Cause", event_causes)
    
    priorities = ['All'] + list(df_all['priority'].unique())
    selected_priority = st.sidebar.selectbox("Priority", priorities)
    
    zones = ['All'] + [z for z in df_all['zone'].unique() if z != 'Unknown']
    selected_zone = st.sidebar.selectbox("Zone", zones)
    
    # Filter Logic
    filtered_df = df_all.copy()
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['event_type'] == selected_type]
    if selected_cause != 'All':
        filtered_df = filtered_df[filtered_df['event_cause'] == selected_cause]
    if selected_priority != 'All':
        filtered_df = filtered_df[filtered_df['priority'] == selected_priority]
    if selected_zone != 'All':
        filtered_df = filtered_df[filtered_df['zone'] == selected_zone]

# Main Panel
st.markdown("<h1 class='title-gradient'>Event-Driven Traffic Management System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Smart City Platform for Event Planning, Congestion Forecasting, and Resource Optimization</p>", unsafe_allow_html=True)

if df_all is None:
    st.info("Upload your dataset or check project configuration to load traffic events data.")
else:
    # Create Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Event Analytics", 
        "🔮 Congestion Risk Forecaster", 
        "👮 Resource Planner",
        "🔀 Diversion Routing"
    ])
    
    # ------------------ Tab 1: Event Analytics ------------------
    with tab1:
        # High Level KPIs
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        total_incidents = len(filtered_df)
        road_closure_pct = (filtered_df['requires_road_closure'].sum() / total_incidents * 100) if total_incidents > 0 else 0.0
        avg_duration = filtered_df['duration_minutes'].mean() if total_incidents > 0 else 0.0
        
        # Calculate active incidents (status == active or status != closed)
        active_incidents = len(filtered_df[filtered_df['status'].isin(['active', 'pending'])])
        
        with kpi_col1:
            st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-val'>{total_incidents:,}</div>
                    <div class='kpi-lbl'>Total Incidents Analyzed</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-val'>{active_incidents:,}</div>
                    <div class='kpi-lbl'>Active/Pending Disruptions</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-val'>{road_closure_pct:.1f}%</div>
                    <div class='kpi-lbl'>Requires Road Closure</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-val'>{avg_duration:.1f}m</div>
                    <div class='kpi-lbl'>Avg Disruption Duration</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # Visualizations Row 1: Hotspot Map & Heatmap
        map_col1, map_col2 = st.columns([2, 1])
        
        with map_col1:
            st.subheader("📍 Incident Hotspots (Geospatial Distribution)")
            # Plotly Mapbox scatter plot
            # Color map for risk levels
            color_discrete_map = {
                'Critical': '#ef4444',
                'High': '#f97316',
                'Medium': '#eab308',
                'Low': '#10b981'
            }
            
            fig_map = px.scatter_mapbox(
                filtered_df,
                lat="latitude",
                lon="longitude",
                color="risk_level",
                color_discrete_map=color_discrete_map,
                size=filtered_df["duration_minutes"].clip(lower=15, upper=240), # Size based on duration
                hover_name="junction",
                hover_data={
                    "event_cause": True,
                    "priority": True,
                    "duration_minutes": ":.1f",
                    "police_station": True,
                    "latitude": False,
                    "longitude": False,
                    "risk_level": False
                },
                zoom=10,
                height=500,
                category_orders={"risk_level": ["Critical", "High", "Medium", "Low"]}
            )
            
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    title="Risk Level",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(15, 23, 42, 0.8)",
                    bordercolor="rgba(51, 65, 85, 0.8)",
                    borderwidth=1,
                    font=dict(color="#f8fafc")
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        with map_col2:
            st.subheader("🔥 Congestion Heatmap")
            # Folium Heatmap
            lat_med = filtered_df['latitude'].median()
            lon_med = filtered_df['longitude'].median()
            if pd.isna(lat_med) or pd.isna(lon_med):
                map_center = [12.9716, 77.5946] # Fallback Bengaluru center
            else:
                map_center = [lat_med, lon_med]
            folium_map = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB dark_matter")
            
            # Extract heat data
            heat_data = filtered_df[['latitude', 'longitude']].dropna().values.tolist()
            if heat_data:
                HeatMap(heat_data, radius=12, blur=15).add_to(folium_map)
                
            st_folium(folium_map, height=500, width=None, key="folium_heatmap")
            
        st.write("")
        st.write("")
        
        # Visualizations Row 2: Temporal & Category Trends
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            st.subheader("⏰ Incident Distribution by Time of Day")
            # Group by hour
            hour_counts = filtered_df['hour_of_day'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['Hour', 'Incident Count']
            
            fig_hour = px.bar(
                hour_counts,
                x='Hour',
                y='Incident Count',
                labels={'Hour': 'Hour of Day (24h)', 'Incident Count': 'Number of Events'},
                color='Incident Count',
                color_continuous_scale='Blues',
                height=350
            )
            fig_hour.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor='#1e293b', tickmode='linear', tick0=0, dtick=2),
                yaxis=dict(gridcolor='#1e293b'),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_hour, use_container_width=True)
            
        with trend_col2:
            st.subheader("📅 Weekly Traffic Incident Patterns")
            # Group by weekday
            day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            day_counts = filtered_df['day_of_week'].value_counts().sort_index().reset_index()
            day_counts.columns = ['Day', 'Incident Count']
            day_counts['DayName'] = day_counts['Day'].map(day_map)
            
            fig_day = px.bar(
                day_counts,
                x='DayName',
                y='Incident Count',
                labels={'DayName': 'Day of Week', 'Incident Count': 'Number of Events'},
                color='Incident Count',
                color_continuous_scale='Purples',
                height=350
            )
            fig_day.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor='#1e293b'),
                yaxis=dict(gridcolor='#1e293b'),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_day, use_container_width=True)
            
        # Visualizations Row 3: Event Cause & Risk Distribution
        cat_col1, cat_col2, cat_col3 = st.columns(3)
        
        with cat_col1:
            st.subheader("🚨 Breakdown of Incident Causes")
            cause_counts = filtered_df['event_cause'].value_counts().reset_index()
            cause_counts.columns = ['Cause', 'Count']
            
            fig_cause = px.pie(
                cause_counts,
                names='Cause',
                values='Count',
                hole=0.4,
                height=350,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cause.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8"),
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(fig_cause, use_container_width=True)
            
        with cat_col2:
            st.subheader("🛡️ Distribution of Congestion Risks")
            risk_counts = filtered_df['risk_level'].value_counts().reindex(['Low', 'Medium', 'High', 'Critical']).fillna(0).reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            
            fig_risk = px.bar(
                risk_counts,
                x='Risk Level',
                y='Count',
                color='Risk Level',
                color_discrete_map=color_discrete_map,
                category_orders={'Risk Level': ['Low', 'Medium', 'High', 'Critical']},
                height=350
            )
            fig_risk.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor='#1e293b'),
                yaxis=dict(gridcolor='#1e293b'),
                showlegend=False
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with cat_col3:
            st.subheader("🏢 Most Vulnerable Zones")
            zone_counts = filtered_df['zone'].value_counts().head(8).reset_index()
            zone_counts.columns = ['Zone', 'Count']
            # Remove unknown zone if present
            zone_counts = zone_counts[zone_counts['Zone'] != 'Unknown']
            
            fig_zone = px.bar(
                zone_counts,
                y='Zone',
                x='Count',
                orientation='h',
                color='Count',
                color_continuous_scale='Viridis',
                height=350
            )
            fig_zone.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor='#1e293b'),
                yaxis=dict(gridcolor='#1e293b'),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_zone, use_container_width=True)
            
    # ------------------ Tab 2: Congestion Risk Forecaster ------------------
    with tab2:
        st.subheader("🔮 Event Congestion Forecast Simulator")
        st.write("Input simulated details of a planned or unplanned traffic event to predict the overall congestion risk and disruption duration.")
        
        # Show warning if falling back
        if is_fallback:
            st.info("⚠️ System is running in fallback calculation mode because ML models are not fully trained. Results are calculated using standard heuristics.")
            
        # Form Container
        with st.form("simulation_form"):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                sim_type = st.selectbox("Event Type", ["unplanned", "planned"])
                sim_cause = st.selectbox("Event Cause", ["vehicle_breakdown", "accident", "tree_fall", "water_logging", "construction", "public_event", "congestion", "pot_holes", "road_conditions", "others"])
                sim_priority = st.selectbox("Priority Level", ["Low", "Medium", "High"])
                
            with col_b:
                sim_road_closure = st.checkbox("Requires Road Closure", value=False)
                # Pre-populated location presets
                location_presets = {
                    "Custom Coordinates": (12.9716, 77.5946, "Unknown", "Unknown"),
                    "Central Silk Board Junction (Hosur Road)": (12.9176, 77.6244, "South Zone 1", "SilkBoardJunc"),
                    "Hebbal Flyover Junction (Bellary Road)": (13.0358, 77.5978, "North Zone 1", "HebbalFlyoverJunc"),
                    "Urvashi Junction (Lalbagh Road)": (12.9556, 77.5857, "Central Zone 2", "UrvashiJunction"),
                    "Peenya 14th Cross Junction (Tumkur Road)": (13.0166, 77.5054, "West Zone 1", "Peenya14thCrossJunc"),
                    "Nayandahalli Junction (Mysore Road)": (12.9446, 77.5274, "West Zone 2", "MysoreRd-RingRdJunc(Nayandanahallii)")
                }
                selected_preset = st.selectbox("Location Preset", list(location_presets.keys()))
                
                # Fetch preset coordinates
                preset_lat, preset_lon, preset_zone, preset_junc = location_presets[selected_preset]
                
                sim_lat = st.number_input("Latitude", value=preset_lat, format="%.6f")
                sim_lon = st.number_input("Longitude", value=preset_lon, format="%.6f")
                
            with col_c:
                sim_start_date = st.date_input("Start Date")
                sim_start_time = st.time_input("Start Time")
                sim_zone = st.text_input("Zone / Corporation Area", value=preset_zone)
                sim_junction = st.text_input("Junction Name", value=preset_junc)
                
            submit_btn = st.form_submit_button("🔮 Predict Congestion & Forecast Impact")
            
        if submit_btn:
            # Fix 4.3: validate coordinates are within Bengaluru bounds
            lat_min, lat_max = BENGALURU_LAT_RANGE
            lon_min, lon_max = BENGALURU_LON_RANGE
            if not (lat_min <= sim_lat <= lat_max and lon_min <= sim_lon <= lon_max):
                st.warning(
                    f"⚠️ Coordinates ({sim_lat:.4f}, {sim_lon:.4f}) are outside the "
                    f"Bengaluru region ({lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E). "
                    f"Density and diversion results may be inaccurate."
                )

            # Combine datetime
            sim_datetime = pd.to_datetime(f"{sim_start_date} {sim_start_time}")

            # Pack input dictionary
            event_input = {
                'event_type': sim_type,
                'event_cause': sim_cause,
                'priority': sim_priority,
                'requires_road_closure': sim_road_closure,
                'start_datetime': sim_datetime,
                'latitude': sim_lat,
                'longitude': sim_lon,
                'zone': sim_zone if sim_zone else "Unknown",
                'junction': sim_junction if sim_junction else "Unknown",
            }

            # Run Prediction
            try:
                if not is_fallback and os.path.exists("models/duration_regressor.joblib"):
                    # ML Mode (uses duration regressor + formula scoring)
                    prediction = predict_event("models", event_input, df_all)
                else:
                    # Heuristic Fallback Mode (Fix 2.1: uses shared functions, no duplication)
                    r_rad = DENSITY_RADIUS_KM / EARTH_RADIUS_KM
                    ref_coords = np.radians(df_all[['latitude', 'longitude']].values)
                    query_coords = np.radians([[sim_lat, sim_lon]])
                    tree = BallTree(ref_coords, metric='haversine')
                    density = int(tree.query_radius(query_coords, r=r_rad, count_only=True)[0])

                    # Estimate duration from historical median
                    matching = df_all[(df_all['event_cause'] == sim_cause) & (df_all['priority'] == sim_priority)]
                    duration_est = matching['duration_minutes'].median() if not matching.empty else \
                                   df_all[df_all['event_cause'] == sim_cause]['duration_minutes'].median()
                    if pd.isna(duration_est):
                        duration_est = 60.0

                    p95_density = df_all['historical_event_density'].quantile(0.95)

                    # Use shared compute_factor_breakdown (Fix 2.1: single source of truth)
                    breakdown = compute_factor_breakdown(
                        requires_road_closure=sim_road_closure,
                        priority=sim_priority,
                        event_cause=sim_cause,
                        event_type=sim_type,
                        duration_minutes=duration_est,
                        historical_event_density=density,
                        p95_density=p95_density,
                    )

                    w = RISK_WEIGHTS
                    risk_score = (
                        w['road_closure'] * breakdown['road_closure_score'] +
                        w['priority']     * breakdown['priority_score'] +
                        w['cause']        * breakdown['cause_score'] +
                        w['duration']     * breakdown['duration_score'] +
                        w['density']      * breakdown['density_score']
                    )
                    from src.risk_engine import _score_to_level
                    prediction = {
                        'risk_score': round(float(risk_score), 1),
                        'risk_level': _score_to_level(risk_score),
                        'predicted_duration_minutes': round(float(duration_est), 1),
                        'factor_breakdown': breakdown,
                    }
                    
                # Save simulation to Session State to share across tabs
                st.session_state['simulated_event'] = event_input
                st.session_state['simulated_prediction'] = prediction
                
                # Display Prediction Results
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    st.markdown("### Forecast Outputs")
                    
                    risk_lvl = prediction['risk_level']
                    badge_class = f"risk-{risk_lvl.lower()}"
                    
                    st.markdown(f"""
                        <div class="risk-badge {badge_class}">
                            Forecasted Congestion Risk Level: {risk_lvl.upper()}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Score gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction['risk_score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Numeric Risk Score (0-100)", 'font': {'size': 16}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickcolor': "#f8fafc"},
                            'bar': {'color': "#3b82f6"},
                            'bgcolor': "#1e293b",
                            'borderwidth': 2,
                            'bordercolor': "#334155",
                            'steps': [
                                {'range': [0, 35], 'color': '#065f46'},
                                {'range': [35, 55], 'color': '#78350f'},
                                {'range': [55, 75], 'color': '#7c2d12'},
                                {'range': [75, 100], 'color': '#991b1b'}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#f8fafc", 'family': "Outfit"},
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=10)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.metric(
                        label="Forecasted Disruption Duration", 
                        value=f"{prediction['predicted_duration_minutes']:.1f} minutes",
                        delta=f"~{prediction['predicted_duration_minutes']/60:.1f} hours"
                    )
                    
                with res_col2:
                    st.markdown("### Risk Factor Decomposition")
                    st.write("Visual breakdown of features causing this risk level:")
                    
                    factors = prediction['factor_breakdown']
                    factor_df = pd.DataFrame({
                        'Factor': [
                            'Road Closure requirement (25% Weight)',
                            'Priority severity (20% Weight)',
                            'Incident type & cause (20% Weight)',
                            'Disruption duration (20% Weight)',
                            'Historical area density (15% Weight)'
                        ],
                        'Weighted Score Contribution': [
                            factors['road_closure_score'] * 0.25,
                            factors['priority_score'] * 0.20,
                            factors['cause_score'] * 0.20,
                            factors['duration_score'] * 0.20,
                            factors['density_score'] * 0.15
                        ],
                        'Raw Component Score (0-100)': [
                            factors['road_closure_score'],
                            factors['priority_score'],
                            factors['cause_score'],
                            factors['duration_score'],
                            factors['density_score']
                        ]
                    })
                    
                    fig_factors = px.bar(
                        factor_df,
                        y='Factor',
                        x='Weighted Score Contribution',
                        hover_data=['Raw Component Score (0-100)'],
                        color='Weighted Score Contribution',
                        color_continuous_scale='Reds',
                        orientation='h',
                        height=250
                    )
                    fig_factors.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#94a3b8"),
                        xaxis=dict(gridcolor='#1e293b', range=[0, 25]),
                        yaxis=dict(gridcolor='#1e293b'),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_factors, use_container_width=True)
                    
                    st.markdown("#### Engine Analytics Explanation:")
                    st.info(
                        f"This incident has a calculated score of **{prediction['risk_score']}**. "
                        f"The primary driver is the **{factor_df.loc[factor_df['Weighted Score Contribution'].idxmax(), 'Factor']}**, "
                        f"which represents the highest contributor to traffic blockages. Proceed to the "
                        f"**Resource Planner** and **Diversion Routing** tabs to deploy traffic response teams."
                    )
                    
            except Exception as e:
                st.error(f"Error predicting congestion risk: {str(e)}")
                
    # ------------------ Tab 3: Resource Planner ------------------
    with tab3:
        st.subheader("👮 Manpower & Barricade Deployment Optimizer")
        
        # Load simulated event if available, otherwise prompt simulator
        if 'simulated_event' in st.session_state:
            sim_event = st.session_state['simulated_event']
            sim_pred = st.session_state['simulated_prediction']
            
            st.success(f"Loaded simulated event: **{sim_event['event_cause']}** in **{sim_event['zone']}** zone (Predicted Risk: **{sim_pred['risk_level']}**)")
            
            # Fetch recommended resources
            rec = recommend_resources(
                risk_level=sim_pred['risk_level'],
                event_cause=sim_event['event_cause'],
                requires_road_closure=sim_event['requires_road_closure']
            )
            
            # Columns
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown("### 📋 Recommended Deployments")
                
                st.markdown(f"""
                    <div class="res-card">
                        <strong>👮 Traffic Police Personnel:</strong> {rec['traffic_personnel_min']} - {rec['traffic_personnel_max']} Officers/Wardens
                        <br><small style="color: #94a3b8;">Deploy officers at entry choke points and diversion exits.</small>
                    </div>
                    <div class="res-card" style="border-left-color: #eab308;">
                        <strong>🚧 Barricades:</strong> {rec['barricades_min']} - {rec['barricades_max']} Units
                        <br><small style="color: #94a3b8;">Use to block closed lanes and isolate the event perimeter.</small>
                    </div>
                    <div class="res-card" style="border-left-color: #10b981;">
                        <strong>🚔 Traffic Patrol & Control Units:</strong> {rec['traffic_control_units']} Interceptor Vehicles
                        <br><small style="color: #94a3b8;">Mobile units for patrolling approach lanes and signaling.</small>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_rec2:
                st.markdown("### 🚑 Emergency Squad Response")
                for item in rec['emergency_response']:
                    st.markdown(f"- **{item}**")
                    
                st.write("")
                # Export Action
                st.subheader("Export Deployment Sheet")
                
                # Create DataFrame for export
                export_data = pd.DataFrame({
                    'Resource Type': ['Traffic Officers (Min)', 'Traffic Officers (Max)', 'Barricades (Min)', 'Barricades (Max)', 'Patrol Interceptors', 'Emergency Services Required'],
                    'Recommendation': [
                        rec['traffic_personnel_min'],
                        rec['traffic_personnel_max'],
                        rec['barricades_min'],
                        rec['barricades_max'],
                        rec['traffic_control_units'],
                        "; ".join(rec['emergency_response'])
                    ]
                })
                
                csv_data = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Deployment Sheet (CSV)",
                    data=csv_data,
                    file_name=f"deployment_plan_{sim_event['junction']}.csv",
                    mime="text/csv"
                )
        else:
            st.info("💡 Please simulate an incident in the **Congestion Risk Forecaster** tab first to see optimized resource recommendations.")
            
    # ------------------ Tab 4: Diversion Routing ------------------
    with tab4:
        st.subheader("🔀 Intelligent Traffic Diversion Coordinator")
        
        if 'simulated_event' in st.session_state:
            sim_event = st.session_state['simulated_event']
            sim_pred = st.session_state['simulated_prediction']
            
            # Fetch diversion suggestion
            div_rec = diversion_engine.recommend_diversion(
                latitude=sim_event['latitude'],
                longitude=sim_event['longitude'],
                zone=sim_event['zone'],
                junction_name=sim_event['junction']
            )
            
            col_div1, col_div2 = st.columns([1, 1])
            
            with col_div1:
                st.markdown("### 🗺️ Diversion Map")
                # Show Mapbox plot of incident and the nearest diversion junctions
                incident_df = pd.DataFrame([{
                    'name': f"INCIDENT: {sim_event['event_cause']}",
                    'lat': sim_event['latitude'],
                    'lon': sim_event['longitude'],
                    'type': 'Incident',
                    'color': '#ef4444',
                    'size': 18
                }])
                
                div_juncs = div_rec['nearest_diversion_junctions']
                juncs_list = []
                for idx, dj in enumerate(div_juncs):
                    juncs_list.append({
                        'name': f"DIVERSION {idx+1}: {dj['junction']}",
                        'lat': dj['latitude'],
                        'lon': dj['longitude'],
                        'type': 'Diversion Junction',
                        'color': '#3b82f6',
                        'size': 12
                    })
                    
                map_df = pd.concat([incident_df, pd.DataFrame(juncs_list)], ignore_index=True)
                
                fig_div_map = px.scatter_mapbox(
                    map_df,
                    lat="lat",
                    lon="lon",
                    color="type",
                    color_discrete_map={'Incident': '#ef4444', 'Diversion Junction': '#3b82f6'},
                    size="size",
                    hover_name="name",
                    zoom=12,
                    height=450
                )
                fig_div_map.update_layout(
                    mapbox_style="carto-darkmatter",
                    margin={"r":0,"t":0,"l":0,"b":0},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        title="Node Type",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(15, 23, 42, 0.8)",
                        bordercolor="rgba(51, 65, 85, 0.8)",
                        borderwidth=1,
                        font=dict(color="#f8fafc")
                    )
                )
                st.plotly_chart(fig_div_map, use_container_width=True)
                
            with col_div2:
                st.markdown("### 📋 Routing Action Strategy")
                
                if div_rec['is_predefined']:
                    st.info("⭐ Standard Predefined Routing Protocol Activated for Major Bottleneck")
                else:
                    st.info("🔍 Spatial Neighbor-Routing Activated")
                    
                st.markdown(f"```\n{div_rec['diversion_plan']}\n```")
                
                st.write("")
                st.markdown("#### Nearest Diversion Nodes:")
                for idx, dj in enumerate(div_juncs):
                    st.markdown(f"**{idx+1}. {dj['junction']}**")
                    st.markdown(f"- Distance: {dj['distance_km']} km away")
                    st.markdown(f"- Coordinates: `{dj['latitude']:.4f}, {dj['longitude']:.4f}`")
                    
        else:
            st.info("💡 Please simulate an incident in the **Congestion Risk Forecaster** tab first to generate routing and diversion suggestions.")

# Dummy WSGI app variable to suppress Vercel deployment error
# NOTE: Streamlit requires a persistent server and cannot run natively on Vercel's serverless functions.
app = None
