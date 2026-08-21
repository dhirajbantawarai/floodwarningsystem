import os
import shutil
import site

#custom_dir = r"C:\P\floodwarningsys\custom_files_new"

custom_dir = os.path.join(os.path.dirname(__file__), "custom_files_new")

site_packages_dirs = site.getsitepackages()

dest_dir = None
dest_dir1 = None

for path in site_packages_dirs:
    if "dist-packages" in path or "site-packages" in path:
        dest_dir = os.path.join(
            path,
            "neuralhydrology",
            "datasetzoo"
        )

        dest_dir1 = os.path.join(
            path,
            "neuralhydrology",
            "utils"
        )

        break


if dest_dir and os.path.exists(dest_dir):

    print(f"NeuralHydrology datasetzoo found at:")
    print(dest_dir)

    print("\nCopying custom files...")

    shutil.copy(
        os.path.join(custom_dir, "camelsgbv2h.py"),
        os.path.join(dest_dir, "camelsgbv2h.py")
    )

    shutil.copy(
        os.path.join(custom_dir, "camelsgbv2.py"),
        os.path.join(dest_dir, "camelsgbv2.py")
    )

    shutil.copy(
        os.path.join(custom_dir, "__init__.py"),
        os.path.join(dest_dir, "__init__.py")
    )

    shutil.copy(
        os.path.join(custom_dir, "config.py"),
        os.path.join(dest_dir1, "config.py")
    )

    print("\nCustom files successfully registered in NeuralHydrology!")

else:

    print(
        "Could not automatically locate "
        "neuralhydrology/datasetzoo in site-packages."
    )