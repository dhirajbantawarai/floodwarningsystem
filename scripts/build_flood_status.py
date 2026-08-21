import pandas as pd


# Load final predictions and thresholds
pred = pd.read_csv(
    "outputs/gnn_normalized_test_predictions.csv"
)

thresholds = pd.read_csv(
    "outputs/flood_warning_thresholds.csv"
)

pred["basin"] = pred["basin"].astype(str)
thresholds["basin"] = thresholds["basin"].astype(str)


# Combine prediction with basin thresholds
df = pred.merge(
    thresholds,
    on="basin",
    how="left"
)


# Discharge cannot be negative
df["prediction"] = df["gnn_prediction"].clip(lower=0)


# Assign flood status
def get_status(row):

    if row["prediction"] >= row["severe"]:
        return "Severe"

    if row["prediction"] >= row["warning"]:
        return "Warning"

    if row["prediction"] >= row["watch"]:
        return "Watch"

    return "Normal"


df["status"] = df.apply(
    get_status,
    axis=1
)


# Save full history
df.to_csv(
    "outputs/flood_warning_status.csv",
    index=False
)


# Save latest date for map
df["date"] = pd.to_datetime(df["date"])

latest_date = df["date"].max()

latest = df[
    df["date"] == latest_date
]

latest.to_csv(
    "outputs/latest_flood_status.csv",
    index=False
)


print("\nFLOOD WARNING STATUS CREATED")
print("----------------------------")

print("Latest date:", latest_date.date())
print("Basins:", len(latest))

print("\nStatus counts:")
print(latest["status"].value_counts())

print("\nSaved:")
print("outputs/flood_warning_status.csv")
print("outputs/latest_flood_status.csv")