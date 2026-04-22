# ============================================
# PHASE 2 — Battery Health Data Pipeline
# ============================================

import pandas as pd
import numpy as np
import os

print("="*50)
print("  EV BATTERY HEALTH PIPELINE")
print("="*50)

# --- STEP A: Load discharge metadata ---
meta = pd.read_csv('discharge_metadata.csv')
meta['capacity'] = pd.to_numeric(meta['capacity'], errors='coerce')
meta = meta.dropna(subset=['capacity'])
print(f"Valid discharge cycles after cleaning: {len(meta)}")
print(f"\nDischarge cycles to process: {len(meta)}")
print(f"Batteries: {sorted(meta['battery_id'].unique())}")

# --- STEP B: Extract features from each discharge cycle ---
# For each discharge cycle (one CSV file) we extract:
# These are the KEY metrics battery engineers look at
#
# 1. capacity_ah     — total charge delivered (Ampere-hours)
#                      THIS is the most important health indicator
#                      A new battery delivers ~2.0 Ah
#                      A degraded battery delivers much less
#
# 2. avg_voltage     — average voltage during discharge
#                      drops as battery ages
#
# 3. min_voltage     — lowest voltage reached
#                      healthy batteries stay above 2.7V
#
# 4. avg_temp        — average temperature during discharge
#                      higher temp = more stress on battery
#
# 5. max_temp        — peak temperature reached
#                      dangerous if too high (thermal runaway risk)
#
# 6. discharge_time  — how long the discharge lasted
#                      shorter = less capacity = older battery
#
# 7. voltage_drop    — difference between start and end voltage
#                      larger drop = more degraded battery
#
# 8. energy_wh       — total energy delivered (Watt-hours)
#                      voltage × current × time integrated

print("\nExtracting features from discharge cycles...")
print("(Processing 2,794 files — this takes 1-2 minutes)")

records = []
processed = 0
errors = 0

for _, row in meta.iterrows():
    try:
        # Load the cycle file
        filepath = f"cleaned_dataset/data/{row['filename']}"
        df = pd.read_csv(filepath)

        # Skip files with too few readings
        if len(df) < 10:
            continue

        # Calculate time duration in seconds
        duration = df['Time'].max() - df['Time'].min()

        # Skip very short cycles (incomplete data)
        if duration < 50:
            continue

        # Calculate capacity using trapezoidal integration
        # Capacity = integral of current over time
        # We use absolute value because current is negative during discharge
        time_hours = df['Time'].values / 3600  # convert seconds to hours
        current_abs = abs(df['Current_measured'].values)
        capacity_ah = np.trapezoid(current_abs, time_hours)

        # Voltage statistics
        avg_voltage  = df['Voltage_measured'].mean()
        min_voltage  = df['Voltage_measured'].min()
        max_voltage  = df['Voltage_measured'].max()
        voltage_drop = df['Voltage_measured'].iloc[0] - df['Voltage_measured'].iloc[-1]

        # Temperature statistics
        avg_temp = df['Temperature_measured'].mean()
        max_temp = df['Temperature_measured'].max()
        temp_rise = df['Temperature_measured'].max() - df['Temperature_measured'].min()

        # Energy delivered (Watt-hours)
        power = abs(df['Voltage_measured'] * df['Current_measured'])
        energy_wh = np.trapezoid(power.values, time_hours)

        # Current statistics
        avg_current = abs(df['Current_measured'].mean())

        records.append({
            'battery_id'    : row['battery_id'],
            'test_id'       : row['test_id'],
            'filename'      : row['filename'],
            'capacity_ah'   : round(capacity_ah, 4),
            'avg_voltage'   : round(avg_voltage, 4),
            'min_voltage'   : round(min_voltage, 4),
            'max_voltage'   : round(max_voltage, 4),
            'voltage_drop'  : round(voltage_drop, 4),
            'avg_temp'      : round(avg_temp, 4),
            'max_temp'      : round(max_temp, 4),
            'temp_rise'     : round(temp_rise, 4),
            'energy_wh'     : round(energy_wh, 4),
            'avg_current'   : round(avg_current, 4),
            'discharge_time': round(duration, 2)
        })
        processed += 1

    except Exception as e:
        errors += 1
        continue

print(f"Processed : {processed} cycles")
print(f"Skipped   : {errors} cycles (incomplete data)")

# --- STEP C: Build the main dataframe ---
df_features = pd.DataFrame(records)

# Sort by battery and cycle order
df_features = df_features.sort_values(
    ['battery_id', 'test_id']
).reset_index(drop=True)

print(f"\nFeature dataset shape: {df_features.shape}")
print(f"\nSample features:")
print(df_features.head())

# --- STEP D: Calculate State of Health (SOH) ---
# SOH = (current capacity / initial capacity) × 100%
# SOH 100% = brand new battery
# SOH 80%  = end of useful life threshold
#            (industry standard — Tesla, BMW both use 80%)
# SOH < 80% = battery should be replaced

print("\nCalculating State of Health (SOH)...")

soh_records = []

for battery_id, group in df_features.groupby('battery_id'):
    group = group.copy().reset_index(drop=True)

    # Initial capacity = first discharge cycle
    initial_capacity = group['capacity_ah'].iloc[0]

    if initial_capacity <= 0:
        continue

    # SOH for each cycle
    group['SOH'] = (group['capacity_ah'] / initial_capacity * 100).clip(0, 100).round(2)

    # Cycle number within this battery
    group['cycle_num'] = range(1, len(group) + 1)

    # Capacity fade = how much capacity lost since new
    group['capacity_fade'] = (
        initial_capacity - group['capacity_ah']
    ).round(4)

    # SOH status label
    group['health_status'] = group['SOH'].apply(
        lambda x: 'Healthy' if x >= 90
        else ('Degrading' if x >= 80 else 'Critical')
    )

    soh_records.append(group)

df_health = pd.concat(soh_records, ignore_index=True)

# --- STEP E: Print health summary ---
print("\n--- Battery Fleet Health Summary ---")
for battery_id, group in df_health.groupby('battery_id'):
    initial = group['capacity_ah'].iloc[0]
    final   = group['capacity_ah'].iloc[-1]
    cycles  = len(group)
    soh     = group['SOH'].iloc[-1]
    status  = group['health_status'].iloc[-1]
    print(f"Battery {battery_id:6} | "
          f"Cycles: {cycles:4} | "
          f"SOH: {soh:6.1f}% | "
          f"Status: {status:10} | "
          f"Initial: {initial:.3f}Ah → Final: {final:.3f}Ah")

# --- STEP F: Save ---
df_health.to_csv('battery_health_data.csv', index=False)
print(f"\nPipeline complete!")
print(f"Total records : {len(df_health)}")
print(f"Batteries     : {df_health['battery_id'].nunique()}")
print(f"\nHealth distribution:")
print(df_health.groupby('health_status')['battery_id'].count())
print("\nSaved to battery_health_data.csv ✅")