import pandas as pd
import numpy as np


df = pd.read_csv(
    "outputs/gnn_normalized_test_predictions.csv"
)


def metrics(obs, pred):

    nse = 1 - np.sum((obs - pred) ** 2) / np.sum(
        (obs - obs.mean()) ** 2
    )

    r = np.corrcoef(obs, pred)[0, 1]

    alpha = pred.std() / obs.std()
    beta = pred.mean() / obs.mean()

    kge = 1 - np.sqrt(
        (r - 1) ** 2 +
        (alpha - 1) ** 2 +
        (beta - 1) ** 2
    )

    rmse = np.sqrt(
        np.mean((obs - pred) ** 2)
    )

    mae = np.mean(
        np.abs(obs - pred)
    )

    return nse, kge, rmse, mae


rows = []


for basin in df["basin"].unique():

    data = df[
        df["basin"] == basin
    ].dropna(
        subset=[
            "observed",
            "nh_prediction",
            "gnn_prediction"
        ]
    )

    if len(data) < 2:
        continue

    obs = data["observed"].values

    nh = metrics(
        obs,
        data["nh_prediction"].values
    )

    gnn = metrics(
        obs,
        data["gnn_prediction"].values
    )

    rows.append({
        "basin": basin,

        "nh_nse": nh[0],
        "gnn_nse": gnn[0],

        "nh_kge": nh[1],
        "gnn_kge": gnn[1],

        "nh_rmse": nh[2],
        "gnn_rmse": gnn[2],

        "nh_mae": nh[3],
        "gnn_mae": gnn[3]
    })


results = pd.DataFrame(rows)


results.to_csv(
    "outputs/final_evaluation_metrics.csv",
    index=False
)


print("\nFINAL MODEL EVALUATION")
print("----------------------")

print("Basins:", len(results))

print("\nMedian NSE")
print("NH:", results["nh_nse"].median())
print("GNN:", results["gnn_nse"].median())

print("\nMedian KGE")
print("NH:", results["nh_kge"].median())
print("GNN:", results["gnn_kge"].median())

print("\nMedian RMSE")
print("NH:", results["nh_rmse"].median())
print("GNN:", results["gnn_rmse"].median())

print("\nMedian MAE")
print("NH:", results["nh_mae"].median())
print("GNN:", results["gnn_mae"].median())

print("\nSaved:")
print("outputs/final_evaluation_metrics.csv")