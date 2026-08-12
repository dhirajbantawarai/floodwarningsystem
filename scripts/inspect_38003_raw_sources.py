from pathlib import Path
import pandas as pd

from neuralhydrology.datasetzoo.camelsgbv2 import (
    load_camels_gb_v2_timeseries
)

DATA_DIR = Path(r"C:\P\floodwarningsys\data")

BASIN = "38003"
WELL = "TL11_9"


# ==========================================================
# 1. CHECK ALL LOADED PRECIPITATION VARIABLES
# ==========================================================

df = load_camels_gb_v2_timeseries(
    data_dir=DATA_DIR,
    basin=BASIN,
    gw_data_frequency="monthly"
)

period = df.loc["2021-08-01":"2021-09-30"]

print("\n====================================")
print("PRECIPITATION AVAILABILITY")
print("====================================")

for column in [
    "precipitation_cehgear",
    "precipitation_haduk"
]:
    if column in period.columns:

        print(f"\n{column}")
        print("Non-NaN:", period[column].notna().sum())
        print("Missing:", period[column].isna().sum())

        if period[column].notna().any():
            print("Mean:", period[column].mean())
            print("Max:", period[column].max())


print("\n====================================")
print("AUG 25 - SEP 10 PRECIPITATION")
print("====================================")

available_columns = [
    col for col in [
        "precipitation_cehgear",
        "precipitation_haduk",
        "gw_groundwater_level",
        "discharge_vol"
    ]
    if col in df.columns
]

print(
    df.loc[
        "2021-08-25":"2021-09-10",
        available_columns
    ].to_string()
)


# ==========================================================
# 2. FIND ORIGINAL MONTHLY GROUNDWATER FILE
# ==========================================================

gw_dir = (
    DATA_DIR
    / "timeseries"
    / "groundwater"
    / "monthly"
)

matches = list(
    gw_dir.glob(f"*{WELL.lower()}*.csv")
)

# Windows glob/case behavior can vary, so use fallback
if not matches:
    matches = [
        file
        for file in gw_dir.glob("*.csv")
        if WELL.lower() in file.name.lower()
    ]

print("\n====================================")
print("GROUNDWATER SOURCE FILE")
print("====================================")

if not matches:
    print("No groundwater file found for:", WELL)

else:
    gw_file = matches[0]

    print("File:")
    print(gw_file)

    raw_gw = pd.read_csv(gw_file)

    print("\nColumns:")
    print(raw_gw.columns.tolist())

    print("\nRaw rows around 2021:")
    print(
        raw_gw[
            raw_gw.astype(str)
            .apply(
                lambda row:
                row.str.contains("2021", na=False).any(),
                axis=1
            )
        ].to_string(index=False)
    )