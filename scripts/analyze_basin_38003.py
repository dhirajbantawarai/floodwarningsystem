from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# SETTINGS
# ==========================================================

RESULT_FILE = Path(
    r"runs\camels_gbv2_gw_multibasin_10_1208_142808"
    r"\test\model_epoch002\test_results.p"
)

BASIN = "38003"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# LOAD NEURALHYDROLOGY TEST RESULTS
# ==========================================================

with open(RESULT_FILE, "rb") as f:
    results = pickle.load(f)

ds = results[BASIN]["1D"]["xr"]


# ==========================================================
# EXTRACT OBSERVED AND PREDICTED DISCHARGE
# ==========================================================

obs = (
    ds["discharge_vol_obs"]
    .isel(time_step=0)
    .to_series()
)

sim = (
    ds["discharge_vol_sim"]
    .isel(time_step=0)
    .to_series()
)


# ==========================================================
# BUILD DATAFRAME
# ==========================================================

df = pd.DataFrame({
    "observed": obs,
    "predicted": sim,
}).dropna()

df.index = pd.to_datetime(df.index)

df["error"] = (
    df["predicted"]
    - df["observed"]
)

df["abs_error"] = df["error"].abs()


# ==========================================================
# BASIC STATISTICS
# ==========================================================

observed_mean = df["observed"].mean()
predicted_mean = df["predicted"].mean()

observed_std = df["observed"].std()
predicted_std = df["predicted"].std()

observed_max = df["observed"].max()
predicted_max = df["predicted"].max()

mae = np.mean(
    np.abs(
        df["observed"]
        - df["predicted"]
    )
)

rmse = np.sqrt(
    np.mean(
        (
            df["observed"]
            - df["predicted"]
        ) ** 2
    )
)

bias = (
    predicted_mean
    - observed_mean
)

correlation = df["observed"].corr(
    df["predicted"]
)

nse = 1 - (
    np.sum(
        (
            df["observed"]
            - df["predicted"]
        ) ** 2
    )
    /
    np.sum(
        (
            df["observed"]
            - observed_mean
        ) ** 2
    )
)


# ==========================================================
# PRINT TOP 10 LARGEST ERRORS
# ==========================================================

print("\n====================================")
print("TOP 10 LARGEST ERRORS")
print("====================================")

top10 = df.nlargest(
    10,
    "abs_error"
)

for date, row in top10.iterrows():

    print(
        f"{date.strftime('%Y-%m-%d')} | "
        f"Observed: {row['observed']:.4f} | "
        f"Predicted: {row['predicted']:.4f} | "
        f"Error: {row['error']:.4f} | "
        f"Abs Error: {row['abs_error']:.4f}"
    )


# ==========================================================
# PRINT BASIN DIAGNOSTICS
# ==========================================================

print("\n====================================")
print(f"BASIN {BASIN} TEST DIAGNOSTICS")
print("====================================")

print(f"Valid rows:       {len(df)}")

print("\nObserved:")
print(f"Mean:             {observed_mean:.4f}")
print(f"Std:              {observed_std:.4f}")
print(f"Maximum:          {observed_max:.4f}")

print("\nPredicted:")
print(f"Mean:             {predicted_mean:.4f}")
print(f"Std:              {predicted_std:.4f}")
print(f"Maximum:          {predicted_max:.4f}")

print("\nErrors:")
print(f"Bias:             {bias:.4f}")
print(f"MAE:              {mae:.4f}")
print(f"RMSE:             {rmse:.4f}")
print(f"Correlation:      {correlation:.4f}")
print(f"Manual NSE:       {nse:.4f}")

print("\nStored NeuralHydrology metrics:")
print(
    f"NSE:              "
    f"{results[BASIN]['1D']['NSE']:.4f}"
)

print(
    f"KGE:              "
    f"{results[BASIN]['1D']['KGE']:.4f}"
)


# ==========================================================
# FIND WORST PREDICTION DATE
# ==========================================================

worst_date = top10.index[0]

worst_observed = top10.iloc[0]["observed"]
worst_predicted = top10.iloc[0]["predicted"]

print("\n====================================")
print("WORST PREDICTION")
print("====================================")

print(
    "Date:",
    worst_date.strftime("%Y-%m-%d")
)

print(
    f"Observed:  {worst_observed:.4f}"
)

print(
    f"Predicted: {worst_predicted:.4f}"
)


# ==========================================================
# SAVE TOP 10 ERRORS AS CSV
# ==========================================================

top10_output = top10.copy()

top10_output.index.name = "date"

top10_file = (
    OUTPUT_DIR
    / f"basin_{BASIN}_top10_errors.csv"
)

top10_output.to_csv(top10_file)

print("\nTop 10 errors saved to:")
print(top10_file)


# ==========================================================
# PLOT FULL TEST PERIOD
# ==========================================================

plt.figure(figsize=(15, 6))

plt.plot(
    df.index,
    df["observed"],
    label="Observed discharge",
    linewidth=1
)

plt.plot(
    df.index,
    df["predicted"],
    label="Predicted discharge",
    linewidth=1
)

# Mark worst prediction
plt.scatter(
    [worst_date],
    [worst_predicted],
    s=80,
    label="Largest prediction error",
    zorder=5
)

plt.xlabel("Date")
plt.ylabel("Discharge volume")

plt.title(
    f"Basin {BASIN} - Observed vs Predicted Discharge\n"
    f"NSE = {results[BASIN]['1D']['NSE']:.3f}, "
    f"KGE = {results[BASIN]['1D']['KGE']:.3f}"
)

plt.legend()

# Keep graph limited to actual test dates
plt.xlim(
    df.index.min(),
    df.index.max()
)

plt.tight_layout()

plot_file = (
    OUTPUT_DIR
    / f"basin_{BASIN}_observed_vs_predicted.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

print("\nPlot saved to:")
print(plot_file)

plt.show()