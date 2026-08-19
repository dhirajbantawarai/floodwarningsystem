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


# Daily NeuralHydrology predictions and residual targets
pred = data.pivot(
    index="date",
    columns="basin",
    values="nh_prediction"
)[basins]

target = data.pivot(
    index="date",
    columns="basin",
    values="residual"
)[basins]


# Training period
train_dates = pred.index[
    pred.index < "2018-10-01"
]


# Simple 2-layer GCN
class GCN(nn.Module):

    def __init__(self):
        super().__init__()

        self.gcn1 = GCNConv(1, 16)
        self.gcn2 = GCNConv(16, 1)

    def forward(self, x):

        x = torch.relu(
            self.gcn1(x, edge_index)
        )

        return self.gcn2(
            x,
            edge_index
        )


model = GCN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Train
print("Nodes:", len(basins))
print("Device:", device)
print("Training days:", len(train_dates))


for epoch in range(1, 11):

    total_loss = 0

    for date in train_dates:

        x = torch.tensor(
            pred.loc[date].values,
            dtype=torch.float32
        ).reshape(-1, 1).to(device)

        y = torch.tensor(
            target.loc[date].values,
            dtype=torch.float32
        ).reshape(-1, 1).to(device)

        # Ignore missing observed values
        mask = ~torch.isnan(y)

        optimizer.zero_grad()

        output = model(x)

        loss = nn.functional.mse_loss(
            output[mask],
            y[mask]
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch}: "
        f"Loss = {total_loss / len(train_dates):.6f}"
    )


# Save model
torch.save(
    model.state_dict(),
    "outputs/gnn_model.pt"
)

print("\nGNN training complete")