
import csv
import json
import re
from pathlib import Path

import cv2
import chromadb


# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

TOPICS_FILE = RAW_DIR / "video_topics.csv"

ANALYSIS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "video_analysis"
    / "video_frame_analysis.json"
)

DATABASE_PATH = BASE_DIR / "vector_database"

COLLECTION_NAME = "coursera_evidence"

MODALITY = "video_visual_analysis"

MAX_IMAGE_DIFFERENCE = 14.0


# ==================================================
# 2. CONVERT TIME TO SECONDS
# ==================================================

def time_to_seconds(value):

    parts = str(value).strip().split(":")

    try:
        numbers = [float(part) for part in parts]

        if len(numbers) == 2:
            minutes, seconds = numbers
            return minutes * 60 + seconds

        if len(numbers) == 3:
            hours, minutes, seconds = numbers
            return hours * 3600 + minutes * 60 + seconds

    except ValueError:
        pass

    raise ValueError(
        f"Invalid timestamp: {value}"
    )


# ==================================================
# 3. LOAD VIDEO TOPICS
# ==================================================

def load_topics():

    if not TOPICS_FILE.exists():
        raise FileNotFoundError(
            f"Topic file not found: {TOPICS_FILE}"
        )

    with TOPICS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        # Ignore empty lines before the CSV header.
        nonempty_lines = (
            line for line in file
            if line.strip()
        )

        reader = csv.DictReader(nonempty_lines)

        if reader.fieldnames is None:
            raise ValueError(
                "video_topics.csv is empty."
            )

        # Remove spaces from header names.
        reader.fieldnames = [
            name.strip()
            for name in reader.fieldnames
        ]

        required_columns = {
            "topic_id",
            "topic_name",
            "approx_start_time",
            "approx_end_time"
        }

        missing_columns = (
            required_columns - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Missing CSV columns: "
                f"{sorted(missing_columns)}. "
                f"Available columns: {reader.fieldnames}"
            )

        topics = []

        for row in reader:

            topic_id = str(
                row.get("topic_id") or ""
            ).strip()

            if not topic_id:
                continue

            start_seconds = time_to_seconds(
                row["approx_start_time"]
            )

            end_seconds = time_to_seconds(
                row["approx_end_time"]
            )

            if end_seconds <= start_seconds:
                print(
                    "Skipping invalid topic range:",
                    topic_id
                )
                continue

            topics.append({
                "topic_id": topic_id,
                "topic_name": str(
                    row.get("topic_name") or topic_id
                ),
                "start": start_seconds,
                "end": end_seconds
            })

    return topics


# ==================================================
# 4. FIND ORIGINAL VIDEO
# ==================================================

def find_video():

    video_files = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.suffix.lower() == ".mp4"
    )

    if len(video_files) != 1:
        raise RuntimeError(
            "Expected exactly one MP4 file "
            f"inside data/raw, but found {len(video_files)}."
        )

    return video_files[0]


# ==================================================
# 5. GET TIMESTAMP CANDIDATE FROM FILENAME
# ==================================================

def get_candidate_timestamp(image_path):

    match = re.search(
        r"_frame_(\d+)\.(jpg|jpeg|png)$",
        image_path.name,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    return int(match.group(1))


# ==================================================
# 6. COMPARE IMAGE TO ORIGINAL VIDEO
# ==================================================

def compare_frame(
    capture,
    image_path,
    timestamp_seconds
):

    saved_image = cv2.imread(
        str(image_path)
    )

    if saved_image is None:
        return False, None

    capture.set(
        cv2.CAP_PROP_POS_MSEC,
        timestamp_seconds * 1000
    )

    success, video_image = capture.read()

    if not success:
        return False, None

    target_size = (320, 180)

    saved_small = cv2.resize(
        saved_image,
        target_size
    )

    video_small = cv2.resize(
        video_image,
        target_size
    )

    difference = cv2.absdiff(
        saved_small,
        video_small
    )

    difference_score = float(
        difference.mean()
    )

    is_match = (
        difference_score <= MAX_IMAGE_DIFFERENCE
    )

    return is_match, difference_score


# ==================================================
# 7. FIND MATCHING TOPIC RANGES
# ==================================================

def find_matching_topics(seconds, topics):

    return [
        topic["topic_id"]
        for topic in topics
        if topic["start"] <= seconds < topic["end"]
    ]


# ==================================================
# 8. MAIN ALIGNMENT WORKFLOW
# ==================================================

def main():

    print("\nVIDEO FRAME TIMESTAMP ALIGNMENT")
    print("=" * 55)

    if not ANALYSIS_FILE.exists():
        raise FileNotFoundError(
            "Video analysis JSON is missing. "
            "Run pipelines/analyze_video_frames.py first."
        )

    records = json.loads(
        ANALYSIS_FILE.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(records, list):
        raise ValueError(
            "Video analysis JSON must contain a list."
        )

    topics = load_topics()

    video_path = find_video()

    print("Original video:", video_path.name)
    print("Visual descriptions:", len(records))
    print("Known topics:", len(topics))

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open the original video."
        )

    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    stored_records = collection.get(
        where={"modality": MODALITY},
        include=["metadatas"]
    )

    # Match filenames with their ChromaDB records.
    stored_by_file = {}

    for record_id, metadata in zip(
        stored_records["ids"],
        stored_records["metadatas"]
    ):

        metadata = metadata or {}

        source_file = metadata.get(
            "source_file"
        )

        if source_file:
            stored_by_file[source_file] = (
                record_id,
                metadata
            )

    checked_count = 0
    verified_count = 0
    updated_count = 0

    try:

        for record in records:

            source_file = str(
                record.get("source_file") or ""
            )

            if not source_file:
                continue

            image_path = (
                BASE_DIR / source_file
            )

            print("\nFRAME:", image_path.name)

            if not image_path.exists():
                print("  Image missing. Skipping.")
                continue

            candidate_seconds = get_candidate_timestamp(
                image_path
            )

            if candidate_seconds is None:
                print(
                    "  No timestamp candidate found."
                )
                continue

            checked_count += 1

            is_match, difference_score = compare_frame(
                capture,
                image_path,
                candidate_seconds
            )

            print(
                "  Candidate timestamp:",
                candidate_seconds,
                "seconds"
            )

            print(
                "  Image difference:",
                difference_score
            )

            if not is_match:
                print(
                    "  Image match failed. "
                    "ChromaDB was not updated."
                )
                continue

            verified_count += 1

            matching_topics = find_matching_topics(
                candidate_seconds,
                topics
            )

            print(
                "  Matching topic ranges:",
                matching_topics
            )

            # Assign a topic only if exactly
            # one approximate range matches.
            if len(matching_topics) == 1:

                topic_id = matching_topics[0]

                topic_status = "unique_range_match"

            elif len(matching_topics) > 1:

                topic_id = "unassigned"

                topic_status = "overlapping_ranges"

            else:

                topic_id = "unassigned"

                topic_status = "no_matching_range"

            if source_file not in stored_by_file:
                print(
                    "  Frame not indexed in ChromaDB."
                )
                continue

            record_id, previous_metadata = (
                stored_by_file[source_file]
            )

            # Preserve existing metadata.
            updated_metadata = dict(
                previous_metadata
            )

            updated_metadata.update({
                "start_seconds": float(
                    candidate_seconds
                ),
                "topic_id": topic_id,
                "topic_candidates": ",".join(
                    matching_topics
                ),
                "timestamp_status": (
                    "candidate_image_match"
                ),
                "topic_mapping_status": topic_status,
                "image_difference": difference_score
            })

            collection.update(
                ids=[record_id],
                metadatas=[updated_metadata]
            )

            updated_count += 1

            print("  ChromaDB updated.")
            print("  Assigned topic:", topic_id)
            print("  Topic status:", topic_status)

    finally:

        capture.release()

    print("\n" + "=" * 55)
    print("ALIGNMENT COMPLETE")

    print(
        "Frames checked:",
        checked_count
    )

    print(
        "Candidate image matches:",
        verified_count
    )

    print(
        "ChromaDB records updated:",
        updated_count
    )

    print(
        "\nNote: Image matching provides evidence "
        "for the candidate timestamp, but does "
        "not establish absolute timing accuracy."
    )

    print(
        "Topic assignments are based on approximate "
        "video topic ranges and may need human review."
    )


# ==================================================
# 9. RUN THE SCRIPT
# ==================================================

if __name__ == "__main__":
    main()
