from pathlib import Path

import pandas as pd

from neuralhydrology.datasetzoo.camelsgbv2 import (
    get_groundwater_mapping_diagnostics,
    load_camels_gb_v2_timeseries,
)

DATA_DIR = Path(r"C:\P\floodwarningsys\data")

TRAIN_START = pd.Timestamp("1970-10-01")
TRAIN_END = pd.Timestamp("2002-09-30")

# Read all CAMELS-GB V2 basin IDs
topographic_file = (
    DATA_DIR
    / "attributes"
    / "camels_gb_v2_topographic_attributes.csv"
)

topo = pd.read_csv(
    topographic_file,
    dtype={"gauge_id": str}
)

all_basins = topo["gauge_id"].astype(str).str.strip().tolist()

print(f"Total CAMELS-GB V2 basins: {len(all_basins)}")
print("Checking monthly groundwater mapping...\n")

diag = get_groundwater_mapping_diagnostics(
    data_dir=DATA_DIR,
    basins=all_basins,
    fallback_max_distance_km=20.0,
    gw_data_frequency="monthly",
)

mapped = diag[diag["mapped_well"] != "NONE"].copy()

print(f"Basins with monthly groundwater mapping: {len(mapped)}")
print("\nChecking train-period groundwater availability...\n")

rows = []

for i, row in mapped.iterrows():

    basin = str(row["basin"])

    try:
        df = load_camels_gb_v2_timeseries(
            data_dir=DATA_DIR,
            basin=basin,
            gw_data_frequency="monthly",
        )

        train = df.loc[
            (df.index >= TRAIN_START)
            & (df.index <= TRAIN_END)
        ]

        gw_rows = (
            int(train["gw_groundwater_level"].notna().sum())
            if "gw_groundwater_level" in train.columns
            else 0
        )

        discharge_rows = (
            int(train["discharge_vol"].notna().sum())
            if "discharge_vol" in train.columns
            else 0
        )

        rows.append(
            {
                "basin": basin,
                "mapped_well": row["mapped_well"],
                "source": row["source"],
                "distance_km": row["distance_km"],
                "gw_train_rows": gw_rows,
                "discharge_train_rows": discharge_rows,
            }
        )

        print(
            f"{basin}: "
            f"GW={gw_rows}, "
            f"discharge={discharge_rows}"
        )

    except Exception as exc:
        print(f"{basin}: ERROR -> {exc}")


results = pd.DataFrame(rows)

# Only keep basins that have both groundwater and discharge
usable = results[
    (results["gw_train_rows"] > 0)
    & (results["discharge_train_rows"] > 365)
].copy()

# Supervisor's run reported these as having insufficient target data,
# so exclude them from our small test.
usable = usable[
    ~usable["basin"].isin(["25029", "46014"])
]

usable = usable.sort_values(
    "gw_train_rows",
    ascending=False,
)

print("\n=================================")
print("TOP 10 MULTI-BASIN CANDIDATES")
print("=================================\n")

print(usable.head(10).to_string(index=False))

selected = usable.head(10)["basin"].tolist()

output_file = Path("basin_lists/multibasin_10.txt")
output_file.parent.mkdir(exist_ok=True)

output_file.write_text(
    "\n".join(selected) + "\n"
)

print("\nSaved basin list to:")
print(output_file)

print("\nSelected basins:")
for basin in selected:
    print(basin)