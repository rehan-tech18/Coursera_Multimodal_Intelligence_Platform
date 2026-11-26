
from pathlib import Path
import pandas as pd

# 1. Locate the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Locate the raw dataset folder
DATA_DIR = BASE_DIR / "data" / "raw"

# 3. Find all CSV files
csv_files = sorted(DATA_DIR.glob("*.csv"))

print("Total CSV files found:", len(csv_files))

# 4. Read and inspect each CSV file
for file in csv_files:

    print("\n" + "=" * 50)
    print("File:", file.name)

    df = pd.read_csv(file)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Column names:", df.columns.tolist())
    print("Missing values:", df.isna().sum().sum())

    print("\nFirst 3 rows:")
    print(df.head(3))
