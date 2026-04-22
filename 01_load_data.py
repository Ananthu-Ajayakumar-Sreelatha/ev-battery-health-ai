
import pandas as pd
import os

print("Loading metadata...")

#Load metadata

meta = pd.read_csv(
    'cleaned_dataset/metadata.csv',
    header=None
)


meta.columns = [
    'test_type', 'timestamp', 'ambient_temp',
    'battery_id', 'test_id', 'uid',
    'filename', 'capacity', 'Re', 'Rct'
]

# Keep only the columns we need
meta = meta[['test_type', 'battery_id',
             'test_id', 'filename', 'capacity']]

print(f"Total records in metadata: {len(meta)}")
print(f"\nTest types found:")
print(meta['test_type'].value_counts())

print(f"\nBatteries found:")
print(meta['battery_id'].unique())



discharge = meta[meta['test_type'] == 'discharge'].copy()
print(f"\nDischarge cycles found: {len(discharge)}")
print(f"Batteries with discharge data: {discharge['battery_id'].nunique()}")


sample_file = discharge['filename'].iloc[0]
sample_path = f"cleaned_dataset/data/{sample_file}"

print(f"\nLoading sample file: {sample_file}")
sample_df = pd.read_csv(sample_path)

print(f"Shape: {sample_df.shape}")
print(f"\nColumns: {list(sample_df.columns)}")
print(f"\nFirst 5 rows:")
print(sample_df.head())
print(f"\nBasic stats:")
print(sample_df.describe())


discharge.to_csv('discharge_metadata.csv', index=False)
print(f"\nDischarge metadata saved!")
print(f"\nData loading complete ✅")