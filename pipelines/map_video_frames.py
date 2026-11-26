
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
FRAMES_DIR = BASE_DIR / "data" / "processed" / "video_frames"

# 1. Load extracted frame information
frames = pd.read_csv(
    FRAMES_DIR / "frame_metadata.csv"
)

# 2. Load video topic information
topics = pd.read_csv(
    RAW_DIR / "video_topics.csv"
)

# 3. Convert HH:MM:SS timestamps into seconds
def time_to_seconds(time_text):
    hours, minutes, seconds = map(
        int, str(time_text).split(":")
    )
    return hours * 3600 + minutes * 60 + seconds

topics["start_seconds"] = (
    topics["approx_start_time"].apply(time_to_seconds)
)

topics["end_seconds"] = (
    topics["approx_end_time"].apply(time_to_seconds)
)

# 4. Match each frame to its lesson topics
records = []

for _, frame in frames.iterrows():

    matching_topics = topics[
        (topics["video_id"] == frame["video_id"])
        & (topics["start_seconds"] <= frame["timestamp_seconds"])
        & (topics["end_seconds"] >= frame["timestamp_seconds"])
    ]

    if matching_topics.empty:
        records.append({
            **frame.to_dict(),
            "topic_id": None,
            "topic_name": None,
            "mapping_status": "unmatched"
        })

    else:
        for _, topic in matching_topics.iterrows():
            records.append({
                **frame.to_dict(),
                "topic_id": topic["topic_id"],
                "topic_name": topic["topic_name"],
                "mapping_status": "approximate"
            })

# 5. Save the mapping
mapping = pd.DataFrame(records)

output_path = FRAMES_DIR / "frame_topic_mapping.csv"

mapping.to_csv(output_path, index=False)

print("Total extracted frames:", len(frames))
print("Total mapping rows:", len(mapping))
print("Matched frames:", mapping.loc[
    mapping["mapping_status"] == "approximate",
    "frame_file"
].nunique())

print("\nSample mappings:")
print(mapping.head(10).to_string(index=False))

print("\nSaved:", output_path)
