from pathlib import Path
import pickle

RESULT_FILE = Path(
    r"runs\camels_gbv2_gw_multibasin_10_1208_142808"
    r"\test\model_epoch002\test_results.p"
)

with open(RESULT_FILE, "rb") as f:
    results = pickle.load(f)

basin = "38003"

daily = results[basin]["1D"]

print("Basin:", basin)
print("1D type:", type(daily))

print("\n==============================")
print("1D CONTENT")
print("==============================")

if isinstance(daily, dict):
    print("Keys:")
    for key, value in daily.items():
        print(f"\nKEY: {key}")
        print("Type:", type(value))

        if hasattr(value, "shape"):
            print("Shape:", value.shape)

        if hasattr(value, "dims"):
            print("Dims:", value.dims)

        if hasattr(value, "data_vars"):
            print("Data variables:")
            for var in value.data_vars:
                print(" -", var)

        print("Preview:")
        try:
            print(value)
        except Exception:
            pass

else:
    print(daily)