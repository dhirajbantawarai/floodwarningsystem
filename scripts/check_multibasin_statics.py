from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\P\floodwarningsys\data")
BASIN_FILE = Path(r"basin_lists\multibasin_10.txt")

STATIC_ATTRIBUTES = [
    "area",
    "elev_mean",
    "sand_perc",
    "silt_perc",
    "clay_perc",
    "p_mean",
    "pet_mean",
]

basins = [
    line.strip()
    for line in BASIN_FILE.read_text().splitlines()
    if line.strip()
]

print("Basins:")
print(basins)
print()

attribute_files = sorted(
    (DATA_DIR / "attributes").glob("*_attributes.csv")
)

frames = []

for file in attribute_files:

    # We already know this is well-level data, not basin-level attributes
    if "groundwaterwell" in file.name:
        continue

    # Hydrometry file is currently renamed .bak, so it is not included anyway

    try:
        df = pd.read_csv(file, dtype={"gauge_id": str})
    except Exception as e:
        print(f"Skipping {file.name}: {e}")
        continue

    if "gauge_id" not in df.columns:
        continue

    df["gauge_id"] = df["gauge_id"].astype(str).str.strip()
    df = df.set_index("gauge_id")

    frames.append(df)

attributes = pd.concat(frames, axis=1)

selected = attributes.loc[
    attributes.index.intersection(basins)
]

print("===================================")
print("SELECTED BASINS FOUND")
print("===================================")
print(f"{len(selected)} / {len(basins)}")
print()

print("===================================")
print("STATIC ATTRIBUTE CHECK")
print("===================================")

problem = False

for attr in STATIC_ATTRIBUTES:

    if attr not in selected.columns:
        print(f"{attr:15} -> MISSING COLUMN")
        problem = True
        continue

    values = pd.to_numeric(selected[attr], errors="coerce")

    missing = int(values.isna().sum())
    std = values.std()
    unique = values.nunique(dropna=True)

    status = "OK"

    if missing > 0:
        status = "HAS NaN"
        problem = True

    if pd.isna(std) or std == 0:
        status = "ZERO/NaN STD"
        problem = True

    print(
        f"{attr:15} "
        f"std={std:.6f} "
        f"unique={unique:2d} "
        f"missing={missing:2d} "
        f"=> {status}"
    )

print()

if problem:
    print(" STATIC ATTRIBUTE CHECK FAILED")
else:
    print(" ALL STATIC ATTRIBUTES ARE SAFE FOR MULTI-BASIN TRAINING")