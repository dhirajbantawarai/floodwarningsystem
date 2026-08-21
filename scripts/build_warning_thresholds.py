from pathlib import Path
import pandas as pd

from neuralhydrology.datasetzoo.camelsgbv2 import (
    load_camels_gb_v2_timeseries
)


DATA = Path("data")
BASINS = Path(
    "basin_lists/final_basins.txt"
).read_text().splitlines()

START = "1970-10-01"
END = "2002-09-30"

rows = []


for basin in BASINS:

    df = load_camels_gb_v2_timeseries(
        data_dir=DATA,
        basin=basin
    )

    discharge = (
        df.loc[START:END, "discharge_vol"]
        .dropna()
    )

    if len(discharge) < 365:
        continue

    rows.append({
        "basin": basin,
        "watch": discharge.quantile(0.90),
        "warning": discharge.quantile(0.95),
        "severe": discharge.quantile(0.99)
    })


thresholds = pd.DataFrame(rows)

thresholds.to_csv(
    "outputs/flood_warning_thresholds.csv",
    index=False
)

print("\nWARNING THRESHOLDS CREATED")
print("--------------------------")
print("Basins:", len(thresholds))

print("\nSample:")
print(thresholds.head())