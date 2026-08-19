import geopandas as gpd


file = "data/boundaries/camels_gb_v2_catchment_boundaries.shp"

gdf = gpd.read_file(file)

print("Catchments:", len(gdf))
print("CRS:", gdf.crs)
print("Columns:", gdf.columns.tolist())

print("\nSample:")
print(gdf.drop(columns="geometry").head())