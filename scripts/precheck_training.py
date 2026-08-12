from pathlib import Path
from neuralhydrology.datasetzoo.camelsgbv2 import (
    prepare_groundwater_training_context
    )
CONFIG = Path("configs/multibasin_10_gw.yml")

ctx = prepare_groundwater_training_context(
    CONFIG,
    runs_dir = Path("runs"),
    print_diagnostics = True,
    auto_disable_gw_on_empty_train = False
                                               )
print("\n=========")
print("PRECHECK COMPLETED!!")
print("=========")

print("Config:", ctx["config_file"])
print("Train basins:", ctx["train_basins"])
print("Basins mapped to groundwater:", ctx["train_with_gw"])
print("Basins with train-period groundwater:", ctx["train_with_gw_data"])
print("GW auto disabled:", ctx["gw_auto_disabled"])