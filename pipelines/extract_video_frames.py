
from pathlib import Path
import csv
import cv2

# 1. Locate project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "video_frames"

# 2. Create the output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Find our MP4 video
video_files = sorted(RAW_DIR.glob("*.mp4"))

if len(video_files) != 1:
    raise ValueError(
        f"Expected 1 MP4 video, found {len(video_files)}"
    )

video_path = video_files[0]
video_id = "PY101_V01"

# 4. Open the video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Unable to open the video file.")

# 5. Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

if fps <= 0 or total_frames <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS or frame count.")

duration = total_frames / fps

print("Video:", video_path.name)
print("Frames per second:", fps)
print("Total frames:", int(total_frames))
print("Duration in seconds:", round(duration, 2))

# 6. Extract one frame every 10 seconds
interval_seconds = 10
records = []

for second in range(0, int(duration) + 1, interval_seconds):

    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)

    success, frame = cap.read()

    if not success:
        print("Could not read frame at", second, "seconds")
        continue

    frame_name = f"{video_id}_frame_{second:05d}.jpg"
    frame_path = OUTPUT_DIR / frame_name

    saved = cv2.imwrite(str(frame_path), frame)

    if not saved:
        print("Could not save:", frame_name)
        continue

    records.append({
        "video_id": video_id,
        "timestamp_seconds": second,
        "frame_file": frame_name,
        "source_video": video_path.name
    })

    print("Saved:", frame_name)

# 7. Release the video file
cap.release()

# 8. Save frame metadata
metadata_path = OUTPUT_DIR / "frame_metadata.csv"

with open(
    metadata_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "video_id",
            "timestamp_seconds",
            "frame_file",
            "source_video"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

print("\nTotal frames saved:", len(records))
print("Metadata saved:", metadata_path)
