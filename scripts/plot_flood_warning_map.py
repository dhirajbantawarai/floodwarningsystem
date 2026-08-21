import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# --------------------------------------------------
# 1. Load catchment boundaries
# --------------------------------------------------

gdf = gpd.read_file(
    "data/boundaries/camels_gb_v2_catchment_boundaries.shp"
)

gdf["ID_STRING"] = gdf["ID_STRING"].astype(str)


# --------------------------------------------------
# 2. Load latest flood status
# --------------------------------------------------

status = pd.read_csv(
    "outputs/latest_flood_status.csv"
)

status["basin"] = status["basin"].astype(str)


# --------------------------------------------------
# 3. Merge flood status with boundaries
# --------------------------------------------------

gdf = gdf.merge(
    status,
    left_on="ID_STRING",
    right_on="basin",
    how="left"
)


# Catchments without prediction are No data
gdf["status"] = gdf["status"].fillna("No data")


# --------------------------------------------------
# 4. Print checks
# --------------------------------------------------

print("\nMerged status counts:")
print(gdf["status"].value_counts())


print("\nSevere basins:")

severe_basins = gdf[
    gdf["status"] == "Severe"
][
    ["ID_STRING", "prediction", "severe"]
]

print(severe_basins)


# --------------------------------------------------
# 5. Flood warning colors
# --------------------------------------------------

colors = {
    "Normal": "green",
    "Watch": "yellow",
    "Warning": "orange",
    "Severe": "red",
    "No data": "lightgray"
}


# --------------------------------------------------
# 6. Create map
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 14)
)


for level, color in colors.items():

    subset = gdf[
        gdf["status"] == level
    ]

    if len(subset) == 0:
        continue

    subset.plot(
        ax=ax,
        color=color,
        edgecolor="black",
        linewidth=0.25
    )


# --------------------------------------------------
# 7. Title
# --------------------------------------------------

ax.set_title(
    "Hybrid Flood Warning Status by Catchment\n"
    "30 September 2022",
    fontsize=15,
    pad=20
)


# Remove map axes
ax.axis("off")


# --------------------------------------------------
# 8. Manual legend
# --------------------------------------------------

legend_items = [

    Patch(
        facecolor="green",
        edgecolor="black",
        label="Normal"
    ),

    Patch(
        facecolor="yellow",
        edgecolor="black",
        label="Watch"
    ),

    Patch(
        facecolor="orange",
        edgecolor="black",
        label="Warning"
    ),

    Patch(
        facecolor="red",
        edgecolor="black",
        label="Severe"
    ),

    Patch(
        facecolor="lightgray",
        edgecolor="black",
        label="No data"
    )
]


ax.legend(
    handles=legend_items,
    title="Flood Status",
    loc="lower left",
    fontsize=10,
    title_fontsize=11
)


# --------------------------------------------------
# 9. Layout
# --------------------------------------------------

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)


# --------------------------------------------------
# 10. Save map
# --------------------------------------------------

plt.savefig(
    "outputs/flood_warning_map.png",
    dpi=300,
    bbox_inches="tight"
)


# Show map
plt.show()


print("\nMAP CREATED")
print("--------------------------")
print("Saved: outputs/flood_warning_map.png")