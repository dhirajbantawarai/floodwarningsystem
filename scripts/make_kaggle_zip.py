from pathlib import Path
import zipfile

data = Path(r"C:\P\floodwarningsys\data")
output = Path(r"C:\P\camels_kaggle.zip")

folders = [
    data / "attributes",
    data / "timeseries" / "hydro-meteorological" / "daily"
]

with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipf:

    for folder in folders:
        for file in folder.rglob("*"):

            if file.is_file():
                path = file.relative_to(data)
                zipf.write(file, path.as_posix())

print("Created:", output)