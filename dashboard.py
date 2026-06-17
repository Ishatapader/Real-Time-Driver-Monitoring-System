import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# CONFIGURATION

CSV_FILE = "driver_telemetry.csv"
EAR_THRESH = 0.20
MAR_THRESH = 0.65
PITCH_THRESH = -15.0

# Web page layout
st.set_page_config(page_title="Driver Telemetry Dashboard", page_icon="🚘", layout="wide")
st.title("🚘 Post-Trip Driver Telemetry Dashboard")
st.markdown("Interactive analysis of driver safety metrics, micro-sleeps, and fatigue indicators.")


#  DATA INGESTION & CACHING

@st.cache_data
def load_data():
    if not os.path.exists(CSV_FILE):
        return None
    try:
        df = pd.read_csv(CSV_FILE)
        if not df.empty:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except Exception:
        return None

df = load_data()

if df is None or df.empty:
    st.warning("⚠️ No telemetry data found. Please run the drowsiness detector first to log some data!")
    st.stop()


#  KPI METRICS

st.markdown("### 📊 Safety Overview")
col1, col2, col3, col4 = st.columns(4)

total_events = len(df)
eye_events = len(df[df['Warning_Type'] == 'eyes'])
yawn_events = len(df[df['Warning_Type'] == 'yawn'])
nod_events = len(df[df['Warning_Type'] == 'nod'])

col1.metric("Total Warnings", total_events)
col2.metric("Micro-sleeps (Eyes)", eye_events, delta="Critical", delta_color="inverse")
col3.metric("Fatigue (Yawns)", yawn_events, delta="Warning", delta_color="off")
col4.metric("Dangerous Nods", nod_events, delta="Critical", delta_color="inverse")

st.divider()


#  INTERACTIVE PLOTS

def create_chart(df, y_column, title, threshold, warning_type, y_label):
    """Helper function to generate interactive Plotly charts."""
    fig = go.Figure()
    
    # Continuous line
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[y_column], mode='lines', name=y_label, line=dict(width=2)))
    
    # Threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold})")
    
    #  Warning triggers with red dots
    warnings = df[df['Warning_Type'] == warning_type]
    fig.add_trace(go.Scatter(x=warnings['Timestamp'], y=warnings[y_column], mode='markers', 
                             name='System Alarm Triggered', marker=dict(color='red', size=10, symbol='x')))
    
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title=y_label, hovermode="x unified", height=350)
    return fig

st.markdown("### 📈 Time-Series Analysis")

# Interactive charts
st.plotly_chart(create_chart(df, 'EAR', 'Eye Aspect Ratio (Drowsiness)', EAR_THRESH, 'eyes', 'EAR Ratio'), use_container_width=True)
st.plotly_chart(create_chart(df, 'MAR', 'Mouth Aspect Ratio (Yawning)', MAR_THRESH, 'yawn', 'MAR Ratio'), use_container_width=True)
st.plotly_chart(create_chart(df, 'Pitch_Delta', 'Head Posture (Nodding)', PITCH_THRESH, 'nod', 'Pitch Delta (Degrees)'), use_container_width=True)