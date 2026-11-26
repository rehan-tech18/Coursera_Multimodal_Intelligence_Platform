
from pathlib import Path
import pandas as pd

# 1. Project locations
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# 2. Read the datasets
quiz = pd.read_csv(DATA_DIR / "quiz_responses.csv")
comments = pd.read_csv(DATA_DIR / "student_comments.csv")
topics = pd.read_csv(DATA_DIR / "video_topics.csv")

# 3. Validate topic IDs before joining
topic_lookup = topics[
    ["course_id", "video_id", "topic_id", "topic_name"]
].drop_duplicates()

if topic_lookup.duplicated(
    ["course_id", "video_id", "topic_id"]
).any():
    raise ValueError("Duplicate topic identifiers found.")

# 4. Aggregate quiz responses
quiz_summary = (
    quiz.groupby(["course_id", "video_id", "topic_id"])
    .agg(
        total_attempts=("is_correct", "size"),
        correct_answers=("is_correct", "sum")
    )
    .reset_index()
)

quiz_summary["incorrect_answers"] = (
    quiz_summary["total_attempts"]
    - quiz_summary["correct_answers"]
)

quiz_summary["incorrect_rate_pct"] = (
    quiz_summary["incorrect_answers"]
    / quiz_summary["total_attempts"] * 100
).round(2)

# 5. Count feedback records by topic
comment_summary = (
    comments.groupby(["course_id", "video_id", "topic_id"])
    .size()
    .reset_index(name="comment_count")
)

# 6. Combine quiz metrics with comment counts
learning_summary = quiz_summary.merge(
    comment_summary,
    on=["course_id", "video_id", "topic_id"],
    how="left",
    validate="one_to_one"
)

learning_summary["comment_count"] = (
    learning_summary["comment_count"]
    .fillna(0)
    .astype(int)
)

# 7. Add readable topic names
learning_summary = learning_summary.merge(
    topic_lookup,
    on=["course_id", "video_id", "topic_id"],
    how="left",
    validate="many_to_one",
    indicator=True
)

if (learning_summary["_merge"] != "both").any():
    raise ValueError("Some quiz topics are missing from video_topics.csv.")

learning_summary = learning_summary.drop(columns="_merge")

# 8. Sort topics by incorrect-answer rate
learning_summary = learning_summary.sort_values(
    by="incorrect_rate_pct",
    ascending=False
)

# 9. Save the report
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

learning_summary.to_csv(
    OUTPUT_DIR / "learning_difficulty_summary.csv",
    index=False
)

print("\nLEARNING DIFFICULTY ANALYSIS")
print(learning_summary.to_string(index=False))

print("\nReport saved successfully!")
