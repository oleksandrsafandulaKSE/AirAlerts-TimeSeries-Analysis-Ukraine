import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import lightgbm as lgb
import json
import time

# ==========================================
# 1. CONFIGURATION & TOGGLES
# ==========================================
st.set_page_config(page_title="Ukraine Air Raid Risk", layout="wide", page_icon="🚨")

# THE MASTER TOGGLE: Change to 'LIVE' when you get your API Key
DATA_MODE = 'STATIC'  # Options: 'STATIC' or 'LIVE'

# Paths (Adjust these to match your local files)
GEOJSON_PATH = 'ukr_admin_boundaries.geojson/ukr_admin1.geojson'
STATIC_DATA_PATH = 'training_matrix.csv'  # Or wherever you saved your Phase 5 output
MODEL_PATH = 'lgbm_risk_model.txt'  # Assuming you saved your model!


# ==========================================
# 2. CACHED RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_model():
    """Loads the trained LightGBM model into memory once."""
    # If you haven't saved your model to a file yet, you can do so in your training script via:
    # model.booster_.save_model('lgbm_risk_model.txt')
    try:
        return lgb.Booster(model_file=MODEL_PATH)
    except Exception as e:
        st.warning("Model file not found. Using a dummy predictor for UI testing.")
        return None


@st.cache_data
def load_geodata():
    """Loads the UN GeoJSON map boundaries."""
    with open(GEOJSON_PATH, 'r') as f:
        return json.load(f)


@st.cache_data
def load_static_baselines(csv_path):
    """Calculates historical baseline risk per region for RRM calibration."""
    try:
        df = pd.read_csv(csv_path)
        # Calculate historical baseline probability (mean of target) per region
        baselines = df.groupby('location_uid')['target'].mean().to_dict()
        return df, baselines
    except Exception:
        st.error(f"Could not load {csv_path}. Please check the path.")
        return pd.DataFrame(), {}


# ==========================================
# 3. LIVE API VS STATIC DATA HANDLER
# ==========================================
def get_current_system_state(df, mode):
    """
    Fetches the 1x3 feature vector for all regions.
    Switches seamlessly between the CSV and the Live API.
    """
    if mode == 'STATIC':
        # Simulate 'Live' by grabbing the absolute latest timestamp in your CSV
        latest_time = df['target_time'].max()
        current_state = df[df['target_time'] == latest_time].copy()
        timestamp_display = str(latest_time)
        return current_state, timestamp_display

    elif mode == 'LIVE':
        # --- FUTURE API IMPLEMENTATION GOES HERE ---
        # 1. response = requests.get('https://api.alerts.in.ua/v1/alerts', headers={'Authorization': 'Bearer YOUR_KEY'})
        # 2. active_alerts = response.json()
        # 3. Run Phase 4 sweep-line math on `active_alerts` to calculate:
        #    - recency_minutes
        #    - countrywide_active
        #    - neighbor_active_count
        # 4. Return as a formatted Pandas DataFrame matching the model's expected inputs

        st.info("Live API pinged successfully. (Placeholder)")
        # Return a dummy dataframe for now so the app doesn't crash if toggled early
        dummy_state = pd.DataFrame()
        return dummy_state, str(pd.Timestamp.utcnow())


# ==========================================
# 4. MAIN DASHBOARD UI (UX OPTIMIZED)
# ==========================================
def main():
    st.title("🚨 Ukraine Spatiotemporal Risk Dashboard")
    st.markdown("Real-time probabilistic forecasting of kinetic threats using AI.")

    # --- UI UPGRADE: Add an explainer for standard users ---
    with st.expander("ℹ️ How to read this map (Click to expand)", expanded=False):
        st.markdown("""
        **What does this map show?**
        This AI model predicts the likelihood of an air raid starting in the next hour.

        **What is the Threat Multiplier?**
        Some regions naturally experience more alerts than others. The **Threat Multiplier** compares the *current* danger level to that region's *normal* everyday baseline. 
        * **1.0x** = Normal everyday risk.
        * **3.0x** = The risk is 3 times higher than usual for this specific region.
        """)
    st.markdown("---")

    # Load Assets
    model = load_model()
    geojson = load_geodata()
    historical_df, baselines = load_static_baselines(STATIC_DATA_PATH)

    if historical_df.empty:
        st.stop()

    # Fetch Data State
    current_state_df, update_time = get_current_system_state(historical_df, DATA_MODE)

    # ==========================================
    # 5. INFERENCE & RISK CALIBRATION
    # ==========================================
    features = ['recency_minutes', 'countrywide_active', 'neighbor_active_count']

    if model:
        current_state_df['raw_prob'] = model.predict(current_state_df[features])
    else:
        current_state_df['raw_prob'] = np.random.uniform(0.01, 0.15, len(current_state_df))

    current_state_df['baseline_prob'] = current_state_df['location_uid'].map(baselines).replace(0, 0.0001)
    current_state_df['rrm'] = current_state_df['raw_prob'] / current_state_df['baseline_prob']

    def categorize_risk(rrm_value):
        if rrm_value <= 1.5:
            return 'Baseline'
        elif rrm_value <= 3.0:
            return 'Elevated'
        elif rrm_value <= 6.0:
            return 'High'
        else:
            return 'Critical'

    current_state_df['risk_tier'] = current_state_df['rrm'].apply(categorize_risk)

    # ==========================================
    # 6. UX TRANSLATIONS & FORMATTING
    # ==========================================
    # Extract real region names from the GeoJSON properties
    # (Assuming standard UN OCHA format: 'adm1_pcode' and 'adm1_en')
    try:
        pcode_to_name = {f['properties']['adm1_pcode']: f['properties']['adm1_en'] for f in geojson['features']}
        current_state_df['Region Name'] = current_state_df['location_uid'].map(pcode_to_name)
    except KeyError:
        # Fallback if your GeoJSON uses different property names
        current_state_df['Region Name'] = current_state_df['location_uid']

    # Create clean, human-readable columns specifically for the UI
    current_state_df['Threat Multiplier'] = current_state_df['rrm'].apply(lambda x: f"{x:.1f}x Normal")
    current_state_df['Probability'] = current_state_df['raw_prob'].apply(lambda x: f"{x * 100:.1f}%")
    current_state_df['Active Neighbors'] = current_state_df['neighbor_active_count']

    # ==========================================
    # 7. FRONTEND RENDERING
    # ==========================================
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Data Source", value=DATA_MODE,
                  delta="System Active" if DATA_MODE == 'LIVE' else "Simulated Data")
    with col2:
        cw_active = current_state_df['countrywide_active'].max() if not current_state_df.empty else 0
        st.metric(label="Countrywide Active Alerts", value=int(cw_active))
    with col3:
        highest_risk = current_state_df.loc[current_state_df['rrm'].idxmax()]
        # Display the real Region Name instead of the P-Code!
        st.metric(label="Highest Risk Region", value=highest_risk['Region Name'],
                  delta=highest_risk['Threat Multiplier'])

    st.subheader(f"Current Threat Map (Last updated: {update_time})")

    # The UX-Optimized Plotly Map
    fig = px.choropleth_mapbox(
        current_state_df,
        geojson=geojson,
        locations='location_uid',
        featureidkey='properties.adm1_pcode',
        color='risk_tier',
        color_discrete_map={
            'Baseline': '#2E8B57',
            'Elevated': '#FFD700',
            'High': '#FF8C00',
            'Critical': '#DC143C'
        },
        category_orders={"risk_tier": ["Baseline", "Elevated", "High", "Critical"]},
        hover_name='Region Name',  # The bold title in the tooltip
        hover_data={
            'location_uid': False,  # Hide the ugly P-Code
            'risk_tier': False,  # Hide the raw tier name (it's obvious from color)
            'Threat Multiplier': True,
            'Probability': True,
            'Active Neighbors': True
        },
        labels={'risk_tier': 'Risk Level'},  # Cleans up the Legend title
        mapbox_style="carto-positron",
        zoom=4.5,
        center={"lat": 48.3794, "lon": 31.1656},
        opacity=0.7
    )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, dragmode="pan")
    st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    main()