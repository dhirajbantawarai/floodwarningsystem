import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GCNConv


device = "cuda" if torch.cuda.is_available() else "cpu"


# Load data
data = pd.read_csv("outputs/gnn_neuralhydrology_outputs.csv")
edges = pd.read_csv("outputs/gnn_edges.csv")
stats = pd.read_csv("outputs/gnn_normalization.csv")

data["date"] = pd.to_datetime(data["date"])
data["basin"] = data["basin"].astype(str)

edges["source"] = edges["source"].astype(str)
edges["target"] = edges["target"].astype(str)

stats["basin"] = stats["basin"].astype(str)


# Basin order
basins = sorted(data["basin"].unique())
basin_id = {b: i for i, b in enumerate(basins)}

stats = stats.set_index("basin").loc[basins]


# Graph
edge_index = torch.tensor([
    [basin_id[b] for b in edges["source"]],
    [basin_id[b] for b in edges["target"]]
], dtype=torch.long).to(device)


# Daily matrices
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


# Normalize NH input
nh_norm = (
    nh - stats["pred_mean"]
) / stats["pred_std"]


# Same GCN
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


model = GCN().to(device)

model.load_state_dict(
    torch.load(
        "outputs/gnn_normalized_model.pt",
        map_location=device
    )
)

model.eval()


# Final untouched test period
test_dates = nh.index[
    nh.index >= "2020-10-01"
]


rows = []


with torch.no_grad():

    for date in test_dates:

        x = torch.tensor(
            nh_norm.loc[date].values,
            dtype=torch.float32
        ).reshape(-1, 1).to(device)

        x = torch.nan_to_num(x)

        # GNN predicts normalized residual
        residual_norm = (
            model(x)
            .cpu()
            .numpy()
            .flatten()
        )

        # Convert residual back to real discharge units
        residual = (
            residual_norm
            * stats["residual_std"].values
            + stats["residual_mean"].values
        )

        nh_pred = nh.loc[date].values

        final_pred = (
            nh_pred + residual
        )

        observed = obs.loc[date].values

        for i, basin in enumerate(basins):

            rows.append({
                "date": date,
                "basin": basin,
                "observed": observed[i],
                "nh_prediction": nh_pred[i],
                "gnn_prediction": final_pred[i]
            })


results = pd.DataFrame(rows)

results.to_csv(
    "outputs/gnn_normalized_test_predictions.csv",
    index=False
)

# NSE
def nse(obs, pred):

    return 1 - (
        np.sum((obs - pred) ** 2)
        /
        np.sum((obs - obs.mean()) ** 2)
    )


scores = []


for basin in basins:

    d = results[
        results["basin"] == basin
    ].dropna()

    if len(d) < 2:
        continue

    observed = d["observed"].values

    scores.append({
        "basin": basin,

        "NH_NSE": nse(
            observed,
            d["nh_prediction"].values
        ),

        "GNN_NSE": nse(
            observed,
            d["gnn_prediction"].values
        )
    })


scores = pd.DataFrame(scores)

scores.to_csv(
    "outputs/gnn_normalized_test_metrics.csv",
    index=False
)


print("\nNORMALIZED GNN TEST")
print("-------------------")

print(
    "NH median NSE:",
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