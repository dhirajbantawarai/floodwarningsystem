from pathlib import Path

import numpy as np
import pandas as pd

from neuralhydrology.datasetzoo.camelsgbv2 import (
    get_groundwater_mapping_diagnostics,
    load_camels_gb_v2_timeseries,
)


# ==========================================================
# SETTINGS
# ==========================================================

DATA_DIR = Path(r"C:\P\floodwarningsys\data")
BASIN_FILE = Path("basin_lists/multibasin_10.txt")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

GW_MONTHLY_DIR = (
    DATA_DIR
    / "timeseries"
    / "groundwater"
    / "monthly"
)


# ==========================================================
# LOAD BASIN LIST
# ==========================================================

basins = [
    line.strip()
    for line in BASIN_FILE.read_text().splitlines()
    if line.strip()
]

print("Basins:")
print(basins)


# ==========================================================
# GET BASIN -> GROUNDWATER WELL MAPPING
# ==========================================================

mapping = get_groundwater_mapping_diagnostics(
    data_dir=DATA_DIR,
    basins=basins,
    fallback_max_distance_km=20.0,
    gw_data_frequency="monthly",
)

print("\n========================================")
print("GROUNDWATER MAPPING")
print("========================================")
print(mapping.to_string(index=False))


# ==========================================================
# PRECIPITATION COMPLETENESS AUDIT
# ==========================================================

precip_rows = []

print("\n========================================")
print("PRECIPITATION COMPLETENESS")
print("========================================")

for basin in basins:

    df = load_camels_gb_v2_timeseries(
        data_dir=DATA_DIR,
        basin=basin,
        gw_data_frequency="monthly",
    )

    for column in [
        "precipitation_cehgear",
        "precipitation_haduk",
    ]:

        if column not in df.columns:
            non_nan = 0
            missing = len(df)
            completeness = 0.0

        else:
            non_nan = int(df[column].notna().sum())
            missing = int(df[column].isna().sum())

            completeness = (
                non_nan / len(df) * 100
                if len(df) > 0
                else 0
            )

        precip_rows.append({
            "basin": basin,
            "variable": column,
            "total_rows": len(df),
            "non_nan": non_nan,
            "missing": missing,
            "completeness_percent": completeness,
        })

        print(
            f"{basin} | "
            f"{column:24} | "
            f"complete={completeness:6.2f}% | "
            f"missing={missing}"
        )


precip_df = pd.DataFrame(precip_rows)

precip_file = (
    OUTPUT_DIR
    / "multibasin_precipitation_completeness.csv"
)

precip_df.to_csv(
    precip_file,
    index=False
)


# ==========================================================
# SIDE-BY-SIDE PRECIPITATION SUMMARY
# ==========================================================

pivot = precip_df.pivot(
    index="basin",
    columns="variable",
    values="completeness_percent",
)

print("\n========================================")
print("PRECIPITATION COMPARISON (%)")
print("========================================")

print(pivot.to_string())


# ==========================================================
# GROUNDWATER ANOMALY AUDIT
# ==========================================================

gw_summary_rows = []
gw_anomaly_rows = []


def find_groundwater_file(well_id):

    matches = [
        file
        for file in GW_MONTHLY_DIR.glob("*.csv")
        if well_id.lower() in file.name.lower()
    ]

    if not matches:
        return None

    return matches[0]


print("\n========================================")
print("GROUNDWATER MONTHLY ANOMALY CHECK")
print("========================================")


for _, mapping_row in mapping.iterrows():

    basin = str(mapping_row["basin"])
    well = str(mapping_row["mapped_well"])

    if well == "NONE":
        continue

    gw_file = find_groundwater_file(well)

    if gw_file is None:
        print(
            f"{basin} -> {well}: "
            f"monthly file NOT FOUND"
        )
        continue

    gw = pd.read_csv(gw_file)

    if (
        "date" not in gw.columns
        or "groundwater_level" not in gw.columns
    ):
        print(
            f"{basin} -> {well}: "
            f"unexpected columns"
        )
        continue

    gw["date"] = pd.to_datetime(
        gw["date"],
        errors="coerce"
    )

    gw["groundwater_level"] = pd.to_numeric(
        gw["groundwater_level"],
        errors="coerce"
    )

    gw = (
        gw
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    values = gw["groundwater_level"]

    valid_values = values.dropna()

    median = valid_values.median()

    # Robust spread estimate
    mad = (
        valid_values
        .sub(median)
        .abs()
        .median()
    )

    gw["previous_level"] = (
        gw["groundwater_level"].shift(1)
    )

    gw["next_level"] = (
        gw["groundwater_level"].shift(-1)
    )

    gw["jump_from_previous"] = (
        gw["groundwater_level"]
        - gw["previous_level"]
    ).abs()

    gw["jump_to_next"] = (
        gw["next_level"]
        - gw["groundwater_level"]
    ).abs()

    # Robust anomaly score.
    # 0.6745 converts MAD to a scale similar to standard deviation.
    if mad and not np.isnan(mad):

        gw["robust_z"] = (
            0.6745
            * (
                gw["groundwater_level"]
                - median
            )
            / mad
        )

    else:
        gw["robust_z"] = np.nan

    # Flag only large deviations for inspection.
    anomalies = gw[
        gw["robust_z"].abs() >= 6
    ].copy()

    gw_summary_rows.append({
        "basin": basin,
        "mapped_well": well,
        "records": len(gw),
        "non_nan": int(valid_values.shape[0]),
        "median": median,
        "mad": mad,
        "minimum": valid_values.min(),
        "maximum": valid_values.max(),
        "flagged_anomalies": len(anomalies),
    })

    print(
        f"\n{basin} -> {well}"
    )

    print(
        f"Median={median:.4f} | "
        f"Min={valid_values.min():.4f} | "
        f"Max={valid_values.max():.4f} | "
        f"Flagged={len(anomalies)}"
    )

    if not anomalies.empty:

        print("Suspicious monthly observations:")

        for _, row in anomalies.iterrows():

            print(
                f"  {row['date'].date()} | "
                f"level={row['groundwater_level']:.4f} | "
                f"prev={row['previous_level']:.4f} | "
                f"next={row['next_level']:.4f} | "
                f"robust_z={row['robust_z']:.2f}"
            )

            gw_anomaly_rows.append({
                "basin": basin,
                "mapped_well": well,
                "date": row["date"],
                "groundwater_level":
                    row["groundwater_level"],
                "previous_level":
                    row["previous_level"],
                "next_level":
                    row["next_level"],
                "jump_from_previous":
                    row["jump_from_previous"],
                "jump_to_next":
                    row["jump_to_next"],
                "robust_z":
                    row["robust_z"],
            })


# ==========================================================
# SAVE GROUNDWATER AUDIT
# ==========================================================

gw_summary_df = pd.DataFrame(
    gw_summary_rows
)

gw_summary_file = (
    OUTPUT_DIR
    / "multibasin_groundwater_summary.csv"
)

gw_summary_df.to_csv(
    gw_summary_file,
    index=False
)


gw_anomaly_df = pd.DataFrame(
    gw_anomaly_rows
)

gw_anomaly_file = (
    OUTPUT_DIR
    / "multibasin_groundwater_anomalies.csv"
)

gw_anomaly_df.to_csv(
    gw_anomaly_file,
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n========================================")
print("AUDIT COMPLETE")
print("========================================")

print("\nPrecipitation report:")
print(precip_file)

print("\nGroundwater summary:")
print(gw_summary_file)

print("\nGroundwater anomalies:")
print(gw_anomaly_file)

print("\n========================================")
print("AVERAGE PRECIPITATION COMPLETENESS")
print("========================================")

average_completeness = (
    precip_df
    .groupby("variable")[
        "completeness_percent"
    ]
    .mean()
)

print(
    average_completeness.to_string()
)