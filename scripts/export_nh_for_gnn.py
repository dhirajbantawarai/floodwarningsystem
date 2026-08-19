import pickle
import pandas as pd
from pathlib import Path


INPUT = Path("outputs/final_evaluation/test_results.p")
OUTPUT = Path("outputs/gnn_neuralhydrology_outputs.csv")


with open(INPUT, "rb") as file:
    results = pickle.load(file)


all_basins = []

for basin, data in results.items():

    if "1D" not in data:
        continue

    ds = data["1D"]["xr"]

    obs = ds["discharge_vol_obs"].isel(time_step=0).to_series()
    pred = ds["discharge_vol_sim"].isel(time_step=0).to_series()

    df = pd.DataFrame({
        "observed": obs,
        "nh_prediction": pred
    })

    df["basin"] = str(basin)

    # Target for GNN
    df["residual"] = (
        df["observed"]
        - df["nh_prediction"]
    )

    df = df.reset_index()

    all_basins.append(df)


df = pd.concat(all_basins, ignore_index=True)

# Prediction is required as GNN input.
# Observed may be missing.
df = df.dropna(subset=["nh_prediction"])

df.to_csv(OUTPUT, index=False)


print("\nGNN DATASET CREATED")
print("-------------------")
print("Rows:", len(df))
print("Basins:", df["basin"].nunique())
print("Start:", df["date"].min())
print("End:", df["date"].max())
print("Missing targets:", df["residual"].isna().sum())