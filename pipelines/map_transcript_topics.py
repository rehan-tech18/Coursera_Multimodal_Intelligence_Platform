
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "transcripts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load transcript segments
transcript_path = (
    RAW_DIR / "transcripts" / "video_transcript.json"
)

with open(transcript_path, "r", encoding="utf-8") as file:
    transcript = json.load(file)

# 2. Load topic timing metadata
topics = pd.read_csv(RAW_DIR / "video_topics.csv")

# 3. Convert HH:MM:SS to seconds
def to_seconds(time_text):
    hours, minutes, seconds = map(
        int, str(time_text).split(":")
    )
    return hours * 3600 + minutes * 60 + seconds

topics["start_seconds"] = topics["approx_start_time"].apply(
    to_seconds
)

topics["end_seconds"] = topics["approx_end_time"].apply(
    to_seconds
)

# 4. Match transcript segments to topics
mapped_segments = []

for segment in transcript:

    segment_start = segment["start_seconds"]
    segment_end = segment["end_seconds"]

    for _, topic in topics.iterrows():

        # Check if the segment overlaps the topic time range
        overlap = (
            segment_start < topic["end_seconds"]
            and segment_end > topic["start_seconds"]
        )

        if overlap and segment["video_id"] == topic["video_id"]:

            mapped_segments.append({
                "segment_id": segment["segment_id"],
                "course_id": topic["course_id"],
                "video_id": segment["video_id"],
                "topic_id": topic["topic_id"],
                "topic_name": topic["topic_name"],
                "start_seconds": segment_start,
                "end_seconds": segment_end,
                "text": segment["text"],
                "source_video": segment["source_video"],
                "mapping_method": "approximate_time_overlap"
            })

# 5. Save normalized results
output_path = OUTPUT_DIR / "mapped_transcript.csv"

mapped_df = pd.DataFrame(mapped_segments)

mapped_df.to_csv(output_path, index=False)

print("\nTRANSCRIPT TOPIC MAPPING COMPLETE")
print("Original transcript segments:", len(transcript))
print("Mapped topic associations:", len(mapped_df))
print("Unique mapped segments:", mapped_df["segment_id"].nunique())
print("\nTopic counts:")
print(mapped_df["topic_name"].value_counts())
print("\nSaved:", output_path)
