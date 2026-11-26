
import os
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Project configuration
BASE_DIR = Path(__file__).resolve().parent.parent
FRAMES_DIR = BASE_DIR / "data" / "processed" / "video_frames"

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing.")

if not model_name:
    raise ValueError("GEMINI_MODEL is missing.")

client = genai.Client(api_key=api_key)

# 2. Load frame metadata and topic mapping
frames = pd.read_csv(FRAMES_DIR / "frame_metadata.csv")

mapping_path = FRAMES_DIR / "frame_topic_mapping.csv"

if not mapping_path.exists():
    raise FileNotFoundError(
        "Run pipelines/map_video_frames.py first."
    )

mapping = pd.read_csv(mapping_path)

# 3. Select only 3 frames for our first test
TEST_LIMIT = 3

frames = frames.sort_values(
    "timestamp_seconds"
).head(TEST_LIMIT)

output_path = FRAMES_DIR / "frame_descriptions.json"

# 4. Load previous results if available
if output_path.exists():
    results = json.loads(
        output_path.read_text(encoding="utf-8")
    )
else:
    results = []

completed = {
    item["frame_file"]
    for item in results
    if item.get("status") == "success"
}

# 5. Analyze each selected frame
for _, frame in frames.iterrows():

    frame_name = frame["frame_file"]

    if frame_name in completed:
        print("Already processed:", frame_name)
        continue

    frame_path = FRAMES_DIR / frame_name

    frame_topics = mapping[
        mapping["frame_file"] == frame_name
    ]

    topic_names = frame_topics[
        "topic_name"
    ].dropna().unique().tolist()

    prompt = """
Analyze this single video frame from a Python lesson.

Describe:
1. What is visibly shown?
2. What Python code or concepts can be identified?
3. Is any execution output or error visible?
4. What might be difficult for a beginner?

Rules:
- Only use information visible in the image.
- Do not invent instructor dialogue.
- Do not claim that students struggled.
- Say when text is unreadable.
- Keep the answer concise.
"""

    print("\nAnalyzing:", frame_name)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=frame_path.read_bytes(),
                    mime_type="image/jpeg"
                )
            ]
        )

        if not response.text:
            raise RuntimeError("No text returned by Gemini.")

        result = {
            "video_id": frame["video_id"],
            "frame_file": frame_name,
            "timestamp_seconds": int(
                frame["timestamp_seconds"]
            ),
            "topic_names": topic_names,
            "description": response.text,
            "model": model_name,
            "status": "success"
        }

    except Exception as error:
        print("Error:", error)

        result = {
            "video_id": frame["video_id"],
            "frame_file": frame_name,
            "timestamp_seconds": int(
                frame["timestamp_seconds"]
            ),
            "topic_names": topic_names,
            "description": None,
            "model": model_name,
            "status": "failed",
            "error": str(error)
        }

    # Replace any earlier result for this frame
    results = [
        item for item in results
        if item["frame_file"] != frame_name
    ]

    results.append(result)

    # Save progress after each frame
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Status:", result["status"])

    # Avoid rapid consecutive API requests
    time.sleep(2)

print("\nBatch finished.")
print("Results saved to:", output_path)
