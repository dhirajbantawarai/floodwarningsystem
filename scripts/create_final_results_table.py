import pandas as pd


results = pd.DataFrame({
    "Metric": [
        "NSE",
        "KGE",
        "RMSE",
        "MAE"
    ],

    "NeuralHydrology": [
        0.8462,
        0.7877,
        1.3333,
        0.6270
    ],

    "NH + GNN": [
        0.8549,
        0.8346,
        1.2991,
        0.6392
    ],

    "Better Model": [
        "NH + GNN",
        "NH + GNN",
        "NH + GNN",
        "NeuralHydrology"
    ]
})


results.to_csv(
    "outputs/final_model_results.csv",
    index=False
)

print(results)

print("\nSaved:")
print("outputs/final_model_results.csv")