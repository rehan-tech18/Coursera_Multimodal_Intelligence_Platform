
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# Load original quiz responses and expected answers
responses = pd.read_csv(DATA_DIR / "quiz_responses.csv")
expected = pd.read_csv(DATA_DIR / "quiz_analysis_expected.csv")

# Calculate metrics at question level
actual = (
    responses.groupby(["question_id", "topic_id"])
    .agg(
        attempts=("is_correct", "size"),
        correct_count=("is_correct", "sum")
    )
    .reset_index()
)

actual["incorrect_count"] = (
    actual["attempts"] - actual["correct_count"]
)

actual["incorrect_rate_pct"] = (
    actual["incorrect_count"] / actual["attempts"] * 100
).round(2)

# Compare calculated values with expected values
comparison = actual.merge(
    expected[
        ["question_id", "topic_id", "attempts",
         "incorrect_count", "incorrect_rate_pct"]
    ],
    on=["question_id", "topic_id"],
    how="outer",
    suffixes=("_actual", "_expected"),
    validate="one_to_one",
    indicator=True
)

comparison["passed"] = (
    comparison["_merge"].eq("both")
    & comparison["attempts_actual"].eq(
        comparison["attempts_expected"]
    )
    & comparison["incorrect_count_actual"].eq(
        comparison["incorrect_count_expected"]
    )
    & (
        comparison["incorrect_rate_pct_actual"]
        - comparison["incorrect_rate_pct_expected"]
    ).abs().le(0.01)
)

print("\nQUIZ ANALYSIS VALIDATION")
print(
    comparison[
        ["question_id", "topic_id", "passed"]
    ].to_string(index=False)
)

passed = int(comparison["passed"].sum())
total = len(comparison)

print(f"\nTests passed: {passed}/{total}")

if comparison["passed"].all():
    print("SUCCESS: All quiz calculations match!")
else:
    print("FAILED: Some calculations do not match.")
    raise SystemExit(1)
