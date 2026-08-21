import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():

    history = pd.read_csv(
    "outputs/gnn_normalized_test_predictions.csv"
)

    history["basin"] = history["basin"].astype(str)
    history["date"] = pd.to_datetime(history["date"])

    # Catchment boundaries
    map_data = gpd.read_file(
        "data/boundaries/camels_gb_v2_catchment_boundaries.shp"
    )

    # Flood prediction/status
    status = pd.read_csv(
        "outputs/latest_flood_status.csv"
    )

    # Make basin IDs the same type
    map_data["ID_STRING"] = map_data["ID_STRING"].astype(str)
    status["basin"] = status["basin"].astype(str)

    # Join predictions with catchments
    map_data = map_data.merge(
        status,
        left_on="ID_STRING",
        right_on="basin",
        how="left"
    )

    # Catchments without predictions
    map_data["status"] = map_data["status"].fillna("No data")

    # Make geometry lighter for web map
    map_data["geometry"] = map_data.geometry.simplify(500)

    # Convert to latitude / longitude
    map_data = map_data.to_crs(epsg=4326)

    return map_data, status, history


map_data, status, history = load_data()


# --------------------------------------------------
# Page
# --------------------------------------------------

st.set_page_config(
    page_title="Flood Warning System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Warning System")

st.write(
    "NeuralHydrology + GNN Flood Prediction"
)

st.subheader("30 September 2022")


# --------------------------------------------------
# Status summary
# --------------------------------------------------

counts = map_data["status"].value_counts()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Normal", counts.get("Normal", 0))
col2.metric("Watch", counts.get("Watch", 0))
col3.metric("Warning", counts.get("Warning", 0))
col4.metric("Severe", counts.get("Severe", 0))
col5.metric("No Data", counts.get("No data", 0))


# --------------------------------------------------
# Map colors
# --------------------------------------------------

colors = {
    "Normal": "green",
    "Watch": "yellow",
    "Warning": "orange",
    "Severe": "red",
    "No data": "gray"
}


st.write(
    "🟢 Normal  |  🟡 Watch  |  🟠 Warning  |  "
    "🔴 Severe  |  ⚪ No data"
)


# --------------------------------------------------
# Create map
# --------------------------------------------------

m = folium.Map(
    location=[54.5, -3],
    zoom_start=6,
    tiles="OpenStreetMap"
)


folium.GeoJson(

    map_data,

    style_function=lambda feature: {

        "fillColor": colors.get(
            feature["properties"]["status"],
            "gray"
        ),

        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.7
    },


    # Show basin/status when hovering
    tooltip=folium.GeoJsonTooltip(

        fields=[
            "ID_STRING",
            "status"
        ],

        aliases=[
            "Basin:",
            "Status:"
        ]
    ),


    # Show details when clicking
    popup=folium.GeoJsonPopup(

        fields=[
            "ID_STRING",
            "status",
            "prediction",
            "watch",
            "warning",
            "severe"
        ],

        aliases=[
            "Basin:",
            "Status:",
            "Predicted Discharge:",
            "Watch Threshold:",
            "Warning Threshold:",
            "Severe Threshold:"
        ]
    )

).add_to(m)


# --------------------------------------------------
# Display map
# --------------------------------------------------

st_folium(
    m,
    height=650,
    use_container_width=True
)

# --------------------------------------------------
# Basin discharge graph
# --------------------------------------------------

st.subheader("📈 Basin Discharge Prediction")

selected_basin = st.selectbox(
    "Select Basin",
    sorted(history["basin"].unique())
)

basin_data = history[
    history["basin"] == selected_basin
].copy()

basin_data = basin_data.set_index("date")


# Get warning thresholds
threshold = status[
    status["basin"] == selected_basin
].iloc[0]

basin_data["Watch Threshold"] = threshold["watch"]
basin_data["Warning Threshold"] = threshold["warning"]
basin_data["Severe Threshold"] = threshold["severe"]


# Data shown on graph
chart_data = basin_data[
    [
        "observed",
        "nh_prediction",
        "gnn_prediction",
        "Watch Threshold",
        "Warning Threshold",
        "Severe Threshold"
    ]
]

chart_data.columns = [
    "Observed",
    "NeuralHydrology",
    "NH + GNN",
    "Watch",
    "Warning",
    "Severe"
]


st.line_chart(
    chart_data,
    height=400
)

# --------------------------------------------------
# Severe alerts
# --------------------------------------------------

st.subheader("🚨 Severe Flood Alerts")


severe = status[
    status["status"] == "Severe"
][
    [
        "basin",
        "prediction",
        "severe"
    ]
]


if len(severe) > 0:

    severe.columns = [
        "Basin",
        "Predicted Discharge",
        "Severe Threshold"
    ]

    st.dataframe(
        severe,
        hide_index=True,
        use_container_width=True
    )

else:

    st.success(
        "No severe flood alerts."
    )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.info(
    "Flood warning levels are based on historical statistical "
    "thresholds and are not official Environment Agency warnings."
)