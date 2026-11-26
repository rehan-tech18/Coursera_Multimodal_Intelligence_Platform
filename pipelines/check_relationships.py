
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# Read the datasets
quiz = pd.read_csv(DATA_DIR / "quiz_responses.csv")
comments = pd.read_csv(DATA_DIR / "student_comments.csv")
topics = pd.read_csv(DATA_DIR / "video_topics.csv")

# Check how many unique topics each dataset contains
print("Unique quiz topics:", quiz["topic_id"].nunique())
print("Unique comment topics:", comments["topic_id"].nunique())
print("Unique video topics:", topics["topic_id"].nunique())

# Find topics mentioned in comments but absent from video topics
unmatched = set(comments["topic_id"]) - set(topics["topic_id"])

print("Unmatched comment topics:", unmatched)

# Count student comments per topic
comment_counts = (
    comments.groupby("topic_id")
    .size()
    .reset_index(name="comment_count")
)

print("\nComments per topic:")
print(comment_counts)
