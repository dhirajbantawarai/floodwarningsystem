from pathlib import Path

from neuralhydrology.datasetzoo.camelsgbv2 import (
    get_groundwater_mapping_diagnostics
)

DATA_DIR = Path(r"C:\P\floodwarningsys\data")

diag = get_groundwater_mapping_diagnostics(
    data_dir=DATA_DIR,
    basins=["76005"],
    fallback_max_distance_km=20.0,
    gw_data_frequency="monthly"
)

print(diag.to_string(index=False))