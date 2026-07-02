"""

Remove all generated files from the project.

Deletes:
    data/processed/*
    models/*
    reports/figures/*
    reports/shap/*
    reports/*.csv

"""

from pathlib import Path


# Directories


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = [
    #PROJECT_ROOT / "data_processed",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "reports" / "figures",
    PROJECT_ROOT / "reports" / "shap",
]


# File Extensions to Remove

EXTENSIONS = {
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".parquet",
    ".pkl",
    ".joblib",
    ".npy",
    ".html",
    ".svg",
    ".txt",
}

# Delete Files

deleted = 0

print("=" * 60)
print("Cleaning project...")
print("=" * 60)

for folder in PATHS:

    if not folder.exists():
        continue

    for file in folder.rglob("*"):

        if file.is_file() and file.suffix.lower() in EXTENSIONS:

            try:
                file.unlink()
                deleted += 1
                print(f"Deleted: {file.relative_to(PROJECT_ROOT)}")

            except Exception as e:
                print(f"Could not delete {file.name}: {e}")

# Remove Empty Directories


for folder in reversed(PATHS):

    if folder.exists():

        for subdir in folder.rglob("*"):

            if subdir.is_dir():

                try:
                    subdir.rmdir()
                except OSError:
                    pass


print("\n" + "=" * 60)
print(f"Finished. Deleted {deleted} generated files.")
print("Raw data was not modified.")
print("=" * 60)