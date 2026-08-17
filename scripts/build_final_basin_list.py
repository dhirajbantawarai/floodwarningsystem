from pathlib import Path
import pandas as pd


DATA = Path(r"C:\P\floodwarningsys\data")
ATTR = DATA / "attributes"
TS = DATA / "timeseries" / "hydro-meteorological" / "daily"

START = "1970-10-01"
END = "2022-09-30"

inputs = [
    "precipitation_haduk",
    "temperature_haduk",
    "pet_hydrope"
]

# ---------------------------------
# Load static attributes
# ---------------------------------

topo = pd.read_csv(
    ATTR / "camels_gb_v2_topographic_attributes.csv",
    dtype={"gauge_id": str}
)[["gauge_id", "area", "elev_mean"]]

soil = pd.read_csv(
    ATTR / "camels_gb_v2_soil_attributes.csv",
    dtype={"gauge_id": str}
)[["gauge_id", "sand_perc", "silt_perc", "clay_perc"]]

climate = pd.read_csv(
    ATTR / "camels_gb_v2_climatic_attributes.csv",
    dtype={"gauge_id": str}
)[["gauge_id", "p_mean", "pet_mean"]]

attrs = (
    topo
    .merge(soil, on="gauge_id")
    .merge(climate, on="gauge_id")
)

# Remove basins with missing static attributes
attrs = attrs.dropna()

accepted = []
rejected = []


# ---------------------------------
# Check each basin
# ---------------------------------

for basin in attrs["gauge_id"]:

    files = list(TS.glob(f"*{basin}*.csv"))

    if not files:
        rejected.append([basin, "No timeseries"])
        continue

    df = pd.read_csv(files[0])

    if "date" not in df.columns:
        rejected.append([basin, "No date"])
        continue

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").loc[START:END]

    required = inputs + ["discharge_vol"]

    if not all(col in df.columns for col in required):
        rejected.append([basin, "Missing variable"])
        continue

    # Require at least 95% complete meteorological data
    input_ok = all(
        df[col].notna().mean() >= 0.95
        for col in inputs
    )

    # Require at least one year of discharge data
    discharge_ok = (
        df["discharge_vol"].notna().sum() >= 365
    )

    if input_ok and discharge_ok:
        accepted.append(basin)
    else:
        rejected.append([basin, "Insufficient data"])


# ---------------------------------
# Save results
# ---------------------------------

Path("basin_lists").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

Path("basin_lists/final_basins.txt").write_text(
    "\n".join(accepted)
)

pd.DataFrame(
    rejected,
    columns=["basin", "reason"]
).to_csv(
    "outputs/final_basin_rejections.csv",
    index=False
)


print("\nFINAL BASIN SELECTION")
print("---------------------")
print("Candidates:", len(attrs))
print("Accepted:  ", len(accepted))
print("Rejected:  ", len(rejected))