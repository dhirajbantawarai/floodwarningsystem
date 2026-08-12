from pathlib import Path

from neuralhydrology.datasetzoo.camelsgbv2 import (
    load_camels_gb_v2_timeseries
)

DATA_DIR = Path(r"C:\P\floodwarningsys\data")
BASIN = "76005"

print(f"Testing basin: {BASIN}")
print(f"Data directory: {DATA_DIR}")

df = load_camels_gb_v2_timeseries(
    data_dir=DATA_DIR,
    basin=BASIN,
    gw_data_frequency="monthly"
)

print("\n=== LOADER SUCCESS ===")
print("Shape:", df.shape)

print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nDate range:")
print(df.index.min(), "to", df.index.max())

print("\nFirst 5 rows:")
print(df.head())

print("\nGroundwater column present:")
print("gw_groundwater_level" in df.columns)

if "gw_groundwater_level" in df.columns:
    print("\nNon-NaN groundwater rows:")
    print(df["gw_groundwater_level"].notna().sum())