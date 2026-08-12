from pathlib import Path
import pandas as pd

ATTR_DIR = Path(r"C:\P\floodwarningsys\data\attributes")

print("Checking attribute CSV files...\n")

for file in sorted(ATTR_DIR.glob("*_attributes.csv")):
    print(f"Checking: {file.name}")

    try:
        df = pd.read_csv(file)
        print(f"  OK -> {df.shape[0]} rows, {df.shape[1]} columns")

    except Exception as e:
        print("  ❌ FAILED")
        print(f"  Error: {e}")

    print()