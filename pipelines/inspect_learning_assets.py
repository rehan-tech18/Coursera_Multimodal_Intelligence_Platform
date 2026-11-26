
from pathlib import Path

# 1. Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Locate our source data
RAW_DIR = BASE_DIR / "data" / "raw"

# 3. Define the file types we want to inspect
file_types = {
    "PDF documents": [".pdf"],
    "PowerPoint slides": [".pptx"],
    "Videos": [".mp4"],
    "Transcripts": [".txt", ".srt", ".vtt"],
    "CSV datasets": [".csv"],
    "Images": [".png", ".jpg", ".jpeg"]
}

print("\nCOURSE ASSET INVENTORY")
print("=" * 50)

# 4. Search all folders inside data/raw
all_files = [
    file for file in RAW_DIR.rglob("*")
    if file.is_file()
]

# 5. Count files by category
for category, extensions in file_types.items():

    matching_files = [
        file for file in all_files
        if file.suffix.lower() in extensions
    ]

    print(f"\n{category}: {len(matching_files)}")

    for file in matching_files:
        print(" -", file.relative_to(RAW_DIR))

print("\nInventory completed.")
