import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GCNConv


device = "cuda" if torch.cuda.is_available() else "cpu"


# Load data
data = pd.read_csv("outputs/gnn_neuralhydrology_outputs.csv")
edges = pd.read_csv("outputs/gnn_edges.csv")

data["date"] = pd.to_datetime(data["date"])
data["basin"] = data["basin"].astype(str)

edges["source"] = edges["source"].astype(str)
edges["target"] = edges["target"].astype(str)


# Basin IDs
basins = sorted(data["basin"].unique())
basin_id = {b: i for i, b in enumerate(basins)}


# Graph connections
edge_index = torch.tensor([
    [basin_id[b] for b in edges["source"]],
    [basin_id[b] for b in edges["target"]]
], dtype=torch.long).to(device)


# Daily data
nh = data.pivot(
    index="date",
    columns="basin",
    values="nh_prediction"
)[basins]

obs = data.pivot(
    index="date",
    columns="basin",
    values="observed"
)[basins]


# Same GCN used during training
class GCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(1, 16)
        self.gcn2 = GCNConv(16, 1)

    def forward(self, x):
        x = torch.relu(
            self.gcn1(x, edge_index)
        )
        return self.gcn2(x, edge_index)


# Load trained model
model = GCN().to(device)

model.load_state_dict(
    torch.load(
        "outputs/gnn_model.pt",
        map_location=device
    )
)

model.eval()


# Test period
test_dates = nh.index[
    nh.index >= "2020-10-01"
]

rows = []


with torch.no_grad():

    for date in test_dates:

        x = torch.tensor(
            nh.loc[date].values,
            dtype=torch.float32
        ).reshape(-1, 1).to(device)

        correction = model(x).cpu().numpy().flatten()

        nh_prediction = nh.loc[date].values
        final_prediction = nh_prediction + correction
        observed = obs.loc[date].values

        for i, basin in enumerate(basins):

            rows.append({
                "date": date,
                "basin": basin,
                "observed": observed[i],
                "nh_prediction": nh_prediction[i],
                "gnn_prediction": final_prediction[i]
            })


results = pd.DataFrame(rows)

results.to_csv(
    "outputs/gnn_test_predictions.csv",
    index=False
)


# NSE function
def nse(observed, predicted):

    return 1 - (
        np.sum((observed - predicted) ** 2)
        /
        np.sum(
            (observed - observed.mean()) ** 2
        )
    )


# Calculate NSE for every basin
scores = []

for basin in basins:

    basin_data = results[
        results["basin"] == basin
    ].dropna()

    if len(basin_data) < 2:
        continue

    observed = basin_data["observed"].values

    nh_score = nse(
        observed,
        basin_data["nh_prediction"].values
    )

    gnn_score = nse(
        observed,
        basin_data["gnn_prediction"].values
    )

    scores.append({
        "basin": basin,
        "NH_NSE": nh_score,
        "GNN_NSE": gnn_score
    })


scores = pd.DataFrame(scores)

scores.to_csv(
    "outputs/gnn_test_metrics.csv",
    index=False
)


print("\nFINAL TEST RESULTS")
print("------------------")

print(
    "NeuralHydrology median NSE:",
    scores["NH_NSE"].median()
)

print(
    "NH + GNN median NSE:",
    scores["GNN_NSE"].median()
)

print(
    "Basins improved:",
    (
        scores["GNN_NSE"]
        > scores["NH_NSE"]
    ).sum()
)

print(
    "Basins evaluated:",
    len(scores)
)