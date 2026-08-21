import pandas as pd
import matplotlib.pyplot as plt


# Load final test scores
df = pd.read_csv(
    "outputs/gnn_normalized_test_metrics.csv"
)

df["improvement"] = (
    df["GNN_NSE"] - df["NH_NSE"]
)


# Summary
print("\nFINAL MODEL COMPARISON")
print("----------------------")
print("Basins:", len(df))
print("NH median NSE:", df["NH_NSE"].median())
print("NH + GNN median NSE:", df["GNN_NSE"].median())
print("Basins improved:", (df["improvement"] > 0).sum())


# Scatter plot
plt.figure(figsize=(8, 8))

plt.scatter(
    df["NH_NSE"],
    df["GNN_NSE"],
    alpha=0.5
)

# Equal-performance line
minimum = min(
    df["NH_NSE"].min(),
    df["GNN_NSE"].min()
)

maximum = max(
    df["NH_NSE"].max(),
    df["GNN_NSE"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)

plt.xlabel("NeuralHydrology NSE")
plt.ylabel("NeuralHydrology + GNN NSE")

plt.title(
    "NeuralHydrology vs Hybrid GNN Performance"
)


plt.xlim(-0.2, 1.05)
plt.ylim(-0.2, 1.05)

plt.tight_layout()

plt.savefig(
    "outputs/final_model_comparison_zoomed.png",
    dpi=200
)

plt.show()


# Save improvement ranking
df.sort_values(
    "improvement",
    ascending=False
).to_csv(
    "outputs/basin_improvement_ranking.csv",
    index=False
)

print("\nSaved:")
print("outputs/final_model_comparison_zoomed.png")
print("outputs/basin_improvement_ranking.csv")