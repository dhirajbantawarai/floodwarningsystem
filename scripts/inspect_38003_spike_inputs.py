from pathlib import Path
import pandas as pd

from neuralhydrology.datasetzoo.camelsgbv2 import (
    load_camels_gb_v2_timeseries
)


# ==========================================================
# SETTINGS
# ==========================================================

DATA_DIR = Path(r"C:\P\floodwarningsys\data")

BASIN = "38003"

START_DATE = "2021-08-01"
END_DATE = "2021-09-30"


# ==========================================================
# LOAD RAW CAMELS-GB V2 DATA
# ==========================================================

df = load_camels_gb_v2_timeseries(
    data_dir=DATA_DIR,
    basin=BASIN,
    gw_data_frequency="monthly"
)


# ==========================================================
# SELECT PERIOD AROUND FALSE FLOOD PEAK
# ==========================================================

window = df.loc[
    START_DATE:END_DATE,
    [
        "precipitation_cehgear",
        "temperature_haduk",
        "pet_hydrope",
        "gw_groundwater_level",
        "discharge_vol"
    ]
].copy()


print("\n====================================")
print("BASIN 38003 - SPIKE INPUT ANALYSIS")
print("====================================")

print(f"Period: {START_DATE} to {END_DATE}")
print(f"Rows: {len(window)}")


# ==========================================================
# BASIC STATISTICS
# ==========================================================

print("\n====================================")
print("INPUT SUMMARY")
print("====================================")

print(
    window.describe().to_string()
)


# ==========================================================
# VALUES AROUND THE WORST PREDICTION
# ==========================================================

print("\n====================================")
print("AUG 25 - SEP 10")
print("====================================")

critical = window.loc[
    "2021-08-25":"2021-09-10"
]

print(
    critical.to_string()
)


# ==========================================================
# HIGHEST RAINFALL DAYS
# ==========================================================

print("\n====================================")
print("TOP 10 PRECIPITATION DAYS")
print("====================================")

top_rain = (
    window
    .sort_values(
        "precipitation_cehgear",
        ascending=False
    )
    .head(10)
)

print(
    top_rain[
        [
            "precipitation_cehgear",
            "gw_groundwater_level",
            "discharge_vol"
        ]
    ].to_string()
)


# ==========================================================
# GROUNDWATER RANGE
# ==========================================================

print("\n====================================")
print("GROUNDWATER")
print("====================================")

gw = window["gw_groundwater_level"]

print(f"Minimum: {gw.min()}")
print(f"Maximum: {gw.max()}")
print(f"Mean:    {gw.mean()}")
print(f"Std:     {gw.std()}")


# ==========================================================
# SAVE FOR LATER ANALYSIS
# ==========================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

output_file = (
    OUTPUT_DIR /
    "basin_38003_spike_inputs.csv"
)

window.to_csv(output_file)

print("\nSaved to:")
print(output_file)