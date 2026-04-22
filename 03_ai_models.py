# ============================================
# PHASE 3 — Battery Health AI Models
# Three techniques:
# 1. Random Forest — predicts remaining range
# 2. DBSCAN — clusters battery behaviour
# 3. Thermal anomaly detection
# ============================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("  EV BATTERY AI MODELS")
print("="*50)

# --- Load data ---
df = pd.read_csv('battery_health_data.csv')
print(f"\nData loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Batteries  : {df['battery_id'].nunique()}")

# -----------------------------------------------
# MODEL 1 — Random Forest: Predict SOH
# -----------------------------------------------
# WHY RANDOM FOREST?
# It's a supervised ML model — meaning it learns
# from labeled examples (we have SOH values)
# It builds 100 decision trees and averages them
# More accurate and robust than a single tree
# Industry standard for battery health prediction
# Used by Tesla and BMW battery management teams
# -----------------------------------------------

print("\n" + "="*50)
print("  MODEL 1: Random Forest — SOH Prediction")
print("="*50)

# Features the model learns from
feature_cols = [
    'avg_voltage',      # voltage tells us energy state
    'min_voltage',      # lowest point = stress indicator
    'voltage_drop',     # how much voltage fell during discharge
    'avg_temp',         # temperature affects degradation
    'max_temp',         # peak heat = stress
    'temp_rise',        # temperature increase during use
    'energy_wh',        # total energy delivered
    'avg_current',      # current draw pattern
    'discharge_time',   # how long it lasted
    'cycle_num'         # which cycle number we're on
]

# Target variable — what we want to predict
target = 'SOH'

# Remove rows with missing values
df_model = df[feature_cols + [target, 'battery_id']].dropna()

X = df_model[feature_cols]
y = df_model[target]

# Split into training and testing sets
# 80% train, 20% test — standard practice
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"\nTraining Random Forest model...")

# Train the model
rf_model = RandomForestRegressor(
    n_estimators=100,   # 100 decision trees
    max_depth=10,       # tree depth limit — prevents overfitting
    random_state=42,
    n_jobs=-1           # use all CPU cores for speed
)

rf_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = rf_model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"Mean Absolute Error : {mae:.2f}%")
print(f"R² Score            : {r2:.4f}")
print(f"Accuracy Meaning    : Model explains {r2*100:.1f}% of SOH variance")

# Feature importance — which sensor matters most?
print(f"\n--- Feature Importance (what the AI learned) ---")
importance = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"{row['feature']:20} {bar} {row['importance']:.4f}")

# Add predictions to dataframe
df.loc[df_model.index, 'SOH_predicted'] = rf_model.predict(X)

# -----------------------------------------------
# MODEL 2 — DBSCAN: Battery Behaviour Clustering
# -----------------------------------------------
# WHY DBSCAN?
# Unlike K-Means, DBSCAN doesn't need you to specify
# the number of clusters in advance
# It finds clusters based on data density
# It also identifies OUTLIERS automatically
# (points that don't belong to any cluster)
# These outliers = unusual battery behaviour patterns
# Perfect for finding batteries behaving abnormally
# -----------------------------------------------

print("\n" + "="*50)
print("  MODEL 2: DBSCAN — Behaviour Clustering")
print("="*50)

# Use key health indicators for clustering
cluster_features = [
    'avg_voltage',
    'avg_temp',
    'capacity_ah',
    'discharge_time',
    'energy_wh'
]

df_cluster = df[cluster_features].dropna()

# Scale data — DBSCAN requires scaled features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

print(f"\nRunning DBSCAN clustering...")

# eps = maximum distance between points in same cluster
# min_samples = minimum points to form a cluster
dbscan = DBSCAN(eps=0.5, min_samples=10)
cluster_labels = dbscan.fit_predict(X_scaled)

# Add cluster labels back
df.loc[df_cluster.index, 'behaviour_cluster'] = cluster_labels

# DBSCAN uses -1 for outliers (anomalies)
n_clusters  = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_outliers  = list(cluster_labels).count(-1)
n_normal    = len(cluster_labels) - n_outliers

print(f"\n--- Clustering Results ---")
print(f"Behaviour clusters found : {n_clusters}")
print(f"Normal behaviour points  : {n_normal}")
print(f"Unusual behaviour points : {n_outliers} (outliers)")
print(f"Outlier rate             : {n_outliers/len(cluster_labels)*100:.1f}%")

# Show what each cluster looks like
print(f"\n--- Cluster Profiles ---")
for cluster_id in sorted(set(cluster_labels)):
    mask = cluster_labels == cluster_id
    cluster_data = df_cluster[mask]
    label = "⚠️ OUTLIER" if cluster_id == -1 else f"Cluster {cluster_id}"
    print(f"\n{label} ({mask.sum()} cycles):")
    print(f"  Avg Voltage    : {cluster_data['avg_voltage'].mean():.3f}V")
    print(f"  Avg Temp       : {cluster_data['avg_temp'].mean():.1f}°C")
    print(f"  Avg Capacity   : {cluster_data['capacity_ah'].mean():.3f}Ah")
    print(f"  Avg Discharge  : {cluster_data['discharge_time'].mean():.0f}s")

# -----------------------------------------------
# MODEL 3 — Thermal Anomaly Detection
# -----------------------------------------------
# WHY THIS MATTERS:
# Thermal runaway is the most dangerous failure mode
# in EV batteries — it can cause fires
# Early detection of abnormal temperature patterns
# is safety critical in automotive applications
# Tesla, BMW, and all EV makers monitor this closely
#
# We flag a cycle as thermally anomalous if:
# 1. Max temperature exceeds safety threshold
# 2. Temperature rise is unusually rapid
# 3. Combined heat + voltage stress pattern
# -----------------------------------------------

print("\n" + "="*50)
print("  MODEL 3: Thermal Anomaly Detection")
print("="*50)

# Calculate thresholds from data statistics
# Using mean + 2 standard deviations
# This flags the top ~2.5% most extreme readings
temp_threshold  = df['max_temp'].mean() + 2 * df['max_temp'].std()
rise_threshold  = df['temp_rise'].mean() + 2 * df['temp_rise'].std()

print(f"\nThermal safety thresholds:")
print(f"Max temperature threshold : {temp_threshold:.1f}°C")
print(f"Temperature rise threshold: {rise_threshold:.1f}°C")

# Flag thermal anomalies
df['thermal_anomaly'] = (
    (df['max_temp'] > temp_threshold) |
    (df['temp_rise'] > rise_threshold)
).astype(int)

thermal_count = df['thermal_anomaly'].sum()
print(f"\nThermal anomalies detected: {thermal_count}")
print(f"Anomaly rate              : {thermal_count/len(df)*100:.1f}%")

# Check if thermal anomalies correlate with poor health
print(f"\n--- Thermal Anomaly vs Battery Health ---")
thermal_soh = df[df['thermal_anomaly'] == 1]['SOH'].mean()
normal_soh  = df[df['thermal_anomaly'] == 0]['SOH'].mean()
print(f"Avg SOH with thermal anomaly    : {thermal_soh:.1f}%")
print(f"Avg SOH without thermal anomaly : {normal_soh:.1f}%")

if thermal_soh < normal_soh:
    print("✅ Thermal anomalies correctly correlate with lower battery health!")
else:
    print("Thermal patterns detected — useful for safety monitoring")

# --- Save all results ---
df.to_csv('battery_ai_results.csv', index=False)
print(f"\n{'='*50}")
print(f"All AI models complete!")
print(f"Results saved to battery_ai_results.csv ✅")
print(f"{'='*50}")