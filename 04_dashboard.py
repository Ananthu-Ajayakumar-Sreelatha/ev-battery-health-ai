import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="EV Battery Health Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv('battery_ai_results.csv')

df = load_data()

# --- Header ---
st.title("⚡ EV Battery Health & Range Intelligence Dashboard")
st.markdown("**AI-powered battery monitoring | NASA Battery Dataset**")
st.markdown("*Random Forest SOH Prediction | DBSCAN Clustering | Thermal Anomaly Detection*")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("🔋 Battery Selector")
battery_ids = sorted(df['battery_id'].unique())
selected_battery = st.sidebar.selectbox("Select Battery", battery_ids)

battery_df = df[df['battery_id'] == selected_battery].copy()
battery_df = battery_df.sort_values('cycle_num').reset_index(drop=True)

# Key metrics
current_soh      = battery_df['SOH'].iloc[-1]
total_cycles     = int(battery_df['cycle_num'].max())
thermal_count    = int(battery_df['thermal_anomaly'].sum())
current_capacity = battery_df['capacity_ah'].iloc[-1]
initial_capacity = battery_df['capacity_ah'].iloc[0]
capacity_fade    = round(initial_capacity - current_capacity, 4)
current_status   = battery_df['health_status'].iloc[-1]
predicted_soh    = battery_df['SOH_predicted'].iloc[-1] if 'SOH_predicted' in battery_df.columns else None

# --- Health Banner ---
if current_soh >= 90:
    st.success(f"✅ BATTERY {selected_battery} — HEALTHY | SOH: {current_soh:.1f}%")
elif current_soh >= 80:
    st.warning(f"🔶 BATTERY {selected_battery} — DEGRADING | SOH: {current_soh:.1f}% | Schedule maintenance")
else:
    st.error(f"⚠️ BATTERY {selected_battery} — CRITICAL | SOH: {current_soh:.1f}% | Replacement recommended")

# --- KPI Cards ---
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("🔋 Current SOH",
              f"{current_soh:.1f}%",
              delta=f"{'Healthy' if current_soh >= 90 else 'Degrading' if current_soh >= 80 else 'Critical'}")

with c2:
    st.metric("🔄 Discharge Cycles",
              f"{total_cycles}",
              delta="Total completed")

with c3:
    st.metric("⚡ Current Capacity",
              f"{current_capacity:.3f} Ah",
              delta=f"-{capacity_fade:.3f} Ah fade")

with c4:
    st.metric("🌡️ Thermal Anomalies",
              f"{thermal_count}",
              delta=f"{thermal_count/total_cycles*100:.1f}% of cycles")

with c5:
    if predicted_soh is not None:
        st.metric("🤖 AI Predicted SOH",
                  f"{predicted_soh:.1f}%",
                  delta=f"Δ {predicted_soh - current_soh:.1f}% vs actual")
    else:
        st.metric("🤖 AI Predicted SOH", "N/A")

st.markdown("---")

# --- Charts Row 1 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 State of Health Degradation")

    fig_soh = go.Figure()

    # Actual SOH
    fig_soh.add_trace(go.Scatter(
        x=battery_df['cycle_num'],
        y=battery_df['SOH'],
        mode='lines',
        name='Actual SOH',
        line=dict(color='steelblue', width=2)
    ))

    # Predicted SOH
    if 'SOH_predicted' in battery_df.columns:
        fig_soh.add_trace(go.Scatter(
            x=battery_df['cycle_num'],
            y=battery_df['SOH_predicted'],
            mode='lines',
            name='AI Predicted SOH',
            line=dict(color='orange', width=2, dash='dash')
        ))

    # Threshold lines
    fig_soh.add_hline(y=80, line_dash="dash",
                      line_color="red",
                      annotation_text="⚠️ End of Life (80%)")
    fig_soh.add_hline(y=90, line_dash="dot",
                      line_color="orange",
                      annotation_text="🔶 Degrading Zone (90%)")

    fig_soh.update_layout(
        xaxis_title="Discharge Cycle",
        yaxis_title="State of Health (%)",
        yaxis=dict(range=[0, 105]),
        height=380
    )
    st.plotly_chart(fig_soh, use_container_width=True)

with col2:
    st.subheader("⚡ Capacity Fade Over Cycles")

    fig_cap = go.Figure()

    fig_cap.add_trace(go.Scatter(
        x=battery_df['cycle_num'],
        y=battery_df['capacity_ah'],
        mode='lines',
        name='Capacity (Ah)',
        line=dict(color='green', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,200,0,0.1)'
    ))

    # Mark thermal anomalies on capacity curve
    thermal_pts = battery_df[battery_df['thermal_anomaly'] == 1]
    fig_cap.add_trace(go.Scatter(
        x=thermal_pts['cycle_num'],
        y=thermal_pts['capacity_ah'],
        mode='markers',
        name=f'Thermal Anomaly ({len(thermal_pts)})',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig_cap.update_layout(
        xaxis_title="Discharge Cycle",
        yaxis_title="Capacity (Ah)",
        height=380
    )
    st.plotly_chart(fig_cap, use_container_width=True)

# --- Charts Row 2 ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("🌡️ Temperature Profile")

    fig_temp = go.Figure()

    fig_temp.add_trace(go.Scatter(
        x=battery_df['cycle_num'],
        y=battery_df['avg_temp'],
        mode='lines',
        name='Avg Temperature',
        line=dict(color='steelblue', width=1.5)
    ))

    fig_temp.add_trace(go.Scatter(
        x=battery_df['cycle_num'],
        y=battery_df['max_temp'],
        mode='lines',
        name='Max Temperature',
        line=dict(color='orange', width=1.5)
    ))

    # Thermal danger threshold
    temp_threshold = df['max_temp'].mean() + 2 * df['max_temp'].std()
    fig_temp.add_hline(
        y=temp_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"⚠️ Danger ({temp_threshold:.0f}°C)"
    )

    fig_temp.update_layout(
        xaxis_title="Discharge Cycle",
        yaxis_title="Temperature (°C)",
        height=350
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with col4:
    st.subheader("⚡ Voltage Behaviour")

    fig_volt = go.Figure()

    fig_volt.add_trace(go.Scatter(
        x=battery_df['cycle_num'],
        y=battery_df['avg_voltage'],
        mode='lines',
        name='Avg Voltage',
        line=dict(color='purple', width=1.5)
    ))

    fig_volt.add_trace(go.Scatter(
        x=battery_df['cycle_num'],
        y=battery_df['min_voltage'],
        mode='lines',
        name='Min Voltage',
        line=dict(color='red', width=1.5, dash='dot')
    ))

    fig_volt.add_hline(
        y=2.7,
        line_dash="dash",
        line_color="red",
        annotation_text="⚠️ Min Safe Voltage (2.7V)"
    )

    fig_volt.update_layout(
        xaxis_title="Discharge Cycle",
        yaxis_title="Voltage (V)",
        height=350
    )
    st.plotly_chart(fig_volt, use_container_width=True)

# --- Behaviour Cluster Analysis ---
st.markdown("---")
st.subheader("🔬 DBSCAN Behaviour Cluster Analysis")

if 'behaviour_cluster' in battery_df.columns:
    cluster_counts = battery_df['behaviour_cluster'].value_counts()

    col5, col6 = st.columns(2)

    with col5:
        fig_cluster = px.pie(
            values=cluster_counts.values,
            names=[f"{'⚠️ Outlier' if c == -1 else f'Cluster {int(c)}'}"
                   for c in cluster_counts.index],
            title=f"Battery {selected_battery} — Behaviour Distribution"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    with col6:
        st.markdown("**What each cluster means:**")
        st.markdown("""
        | Cluster | Behaviour Pattern |
        |---------|------------------|
        | 0 | Standard discharge — normal operating range |
        | 1-2 | Long discharge — high capacity cycles |
        | 3 | High temperature — stress conditions |
        | 9-10 | Low temperature — cold weather operation |
        | ⚠️ Outlier | Unusual pattern — requires inspection |
        """)

# --- Fleet Overview ---
st.markdown("---")
st.subheader("🚗 Full Battery Fleet Overview")

fleet = df.groupby('battery_id').agg(
    Total_Cycles  =('cycle_num', 'max'),
    Final_SOH     =('SOH', 'last'),
    Avg_Capacity  =('capacity_ah', 'mean'),
    Thermal_Alerts=('thermal_anomaly', 'sum'),
    Outlier_Cycles=('behaviour_cluster', lambda x: (x == -1).sum())
).reset_index()

fleet.columns = [
    'Battery', 'Cycles', 'SOH (%)',
    'Avg Capacity (Ah)', 'Thermal Alerts', 'Outlier Cycles'
]

fleet['SOH (%)'] = fleet['SOH (%)'].round(1)
fleet['Avg Capacity (Ah)'] = fleet['Avg Capacity (Ah)'].round(3)

fleet['Status'] = fleet['SOH (%)'].apply(
    lambda x: '🟢 Healthy' if x >= 90
    else ('🟡 Degrading' if x >= 80 else '🔴 Critical')
)

# Fleet KPIs
f1, f2, f3, f4 = st.columns(4)
f1.metric("Total Batteries", len(fleet))
f2.metric("🟢 Healthy",  len(fleet[fleet['Status'] == '🟢 Healthy']))
f3.metric("🟡 Degrading", len(fleet[fleet['Status'] == '🟡 Degrading']))
f4.metric("🔴 Critical",  len(fleet[fleet['Status'] == '🔴 Critical']))

st.dataframe(
    fleet.sort_values('SOH (%)'),
    use_container_width=True,
    height=400
)

# --- Footer ---
st.markdown("---")
st.caption(
    "⚡ EV Battery Health Dashboard | "
    "Dataset: NASA Battery Dataset | "
    "AI: Random Forest + DBSCAN + Thermal Detection | "
    "Stack: Python · Streamlit · Plotly · scikit-learn"
)