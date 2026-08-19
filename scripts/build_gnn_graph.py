import numpy as np
import pandas as pd
import geopandas as gpd


# Load usable NeuralHydrology basins
nh = pd.read_csv(
    "outputs/gnn_neuralhydrology_outputs.csv",
    usecols=["basin"]
)

basins = set(nh["basin"].astype(str).unique())


# Load catchment locations
gdf = gpd.read_file(
    "data/boundaries/camels_gb_v2_catchment_boundaries.shp"
)

gdf["ID_STRING"] = gdf["ID_STRING"].astype(str)

# Keep only basins available from NeuralHydrology
gdf = gdf[
    gdf["ID_STRING"].isin(basins)
].copy()

gdf = gdf.reset_index(drop=True)


# Build 5-nearest-neighbour graph
edges = []

for i, basin in gdf.iterrows():

    dx = gdf["ceast"] - basin["ceast"]
    dy = gdf["cnorth"] - basin["cnorth"]

    distance = np.sqrt(dx**2 + dy**2)

    nearest = (
        distance
        .sort_values()
        .iloc[1:6]
        .index
    )

    for j in nearest:
        edges.append([
            basin["ID_STRING"],
            gdf.loc[j, "ID_STRING"],
            distance[j] / 1000
        ])


# Save graph
edges = pd.DataFrame(
    edges,
    columns=[
        "source",
        "target",
        "distance_km"
    ]
)

edges.to_csv(
    "outputs/gnn_edges.csv",
    index=False
)


print("\nGNN GRAPH CREATED")
print("-----------------")
print("Nodes:", len(gdf))
print("Edges:", len(edges))
print("Average neighbours:", len(edges) / len(gdf))

print("\nSample:")
print(edges.head())