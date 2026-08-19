from pathlib import Path
import zipfile

root = Path(r"C:\P\floodwarningsys")
output = Path(r"C:\P\flood_code.zip")

files = [
    "custom_files_new/__init__.py",
    "custom_files_new/camelsgbv2.py",
    "custom_files_new/camelsgbv2h.py",
    "custom_files_new/config.py",
    "configs/final_666_haduk.yml",
    "basin_lists/final_basins.txt",
    "install_custom_nh.py",
]

with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for file in files:
        z.write(root / file, file)

print("Created:", output)