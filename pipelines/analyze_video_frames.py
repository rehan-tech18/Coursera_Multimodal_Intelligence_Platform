
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

FRAMES_DIR = (
    BASE_DIR / "data" / "processed" / "video_frames"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" / "video_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "video_frame_analysis.json"

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)

BACKUP_MODEL = "gemini-3.1-flash-lite"

MAX_FRAMES = 5


# ==========================================
# 2. FIND VIDEO FRAMES
# ==========================================

def get_video_frames():

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"Video frames folder not found: {FRAMES_DIR}"
        )

    frames = sorted(
        file
        for file in FRAMES_DIR.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    return frames


# ==========================================
# 3. CHOOSE SAMPLE FRAMES
# ==========================================

def select_sample_frames(frames, limit=5):

    if len(frames) <= limit:
        return frames

    # Choose samples from different positions
    # instead of selecting only the first frames.
    indices = [
        round(
            i * (len(frames) - 1) / (limit - 1)
        )
        for i in range(limit)
    ]

    return [frames[index] for index in indices]


# ==========================================
# 4. ANALYSE ONE IMAGE WITH GEMINI
# ==========================================

def analyse_frame(client, frame_path):

    mime_type = (
        "image/png"
        if frame_path.suffix.lower() == ".png"
        else "image/jpeg"
    )

    image_bytes = frame_path.read_bytes()

    prompt = """
You are analysing a screenshot extracted
from a beginner Python teaching video.

Describe only what is visibly present.

Provide these sections:

1. Visible Content
   Describe text, code, diagrams or interface
   elements that are clearly visible.

2. Python Concepts
   Identify Python concepts shown in the image.

3. Visible Code
   Transcribe short Python snippets if readable.
   Do not invent missing or unclear code.

4. Teaching Value
   Explain what this frame could help students
   understand.

5. Limitations
   Mention anything too small, blurry or
   unclear to interpret confidently.

Important:
Do not assume the instructor said something
unless it is visible in the screenshot.
Do not invent quiz results or learner difficulties.
"""

    models = list(
        dict.fromkeys([MODEL_NAME, BACKUP_MODEL])
    )

    last_error = None

    for model in models:

        try:

            response = client.models.generate_content(
                model=model,
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    )
                ]
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty description."
                )

            return response.text, model

        except Exception as error:

            last_error = error

            if getattr(error, "code", None) != 503:
                raise

            print(
                f"Model {model} temporarily unavailable."
            )

    raise RuntimeError(
        "Gemini could not analyse the frame."
    ) from last_error


# ==========================================
# 5. LOAD PREVIOUS RESULTS
# ==========================================

def load_previous_results():

    if not OUTPUT_FILE.exists():
        return []

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data if isinstance(data, list) else []


# ==========================================
# 6. SAVE RESULTS
# ==========================================

def save_results(results):

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    temporary_file.replace(OUTPUT_FILE)


# ==========================================
# 7. MAIN PROGRAM
# ==========================================

def main():

    print("\nVIDEO FRAME VISUAL ANALYSIS")
    print("=" * 55)

    frames = get_video_frames()

    print("Available frames:", len(frames))

    if not frames:
        print("No images found.")
        return

    selected_frames = select_sample_frames(
        frames,
        limit=MAX_FRAMES
    )

    print("Frames selected:", len(selected_frames))

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from .env"
        )

    results = load_previous_results()

    completed_files = {
        record["source_file"]
        for record in results
    }

    client = genai.Client(api_key=api_key)

    try:

        for number, frame_path in enumerate(
            selected_frames,
            start=1
        ):

            relative_path = (
                frame_path
                .relative_to(BASE_DIR)
                .as_posix()
            )

            print(
                f"\nFrame {number}/{len(selected_frames)}"
            )

            print("File:", frame_path.name)

            if relative_path in completed_files:
                print("Already analysed. Skipping.")
                continue

            try:

                description, model = analyse_frame(
                    client,
                    frame_path
                )

                record = {
                    "course_id": "PY101",
                    "video_id": "PY101_V01",
                    "source_type": "video_frame",
                    "source_file": relative_path,
                    "description": description,
                    "model": model,
                    "analysis_status": "completed",
                    "timestamp_seconds": None,
                    "provenance": "gemini_visual_analysis"
                }

                results.append(record)

                # Save each successful frame immediately.
                save_results(results)

                print("Analysis saved successfully.")
                print("Description preview:")
                print(description[:350])

            except Exception as error:

                print("Frame analysis failed:", error)
                print("Continuing to the next frame.")

    finally:
        client.close()

    print("\n" + "=" * 55)
    print("VIDEO FRAME ANALYSIS FINISHED")
    print("Total saved records:", len(results))
    print("Output file:", OUTPUT_FILE)

    print(
        "\nNote: Visual descriptions are AI-generated "
        "and have not yet been human-verified."
    )


if __name__ == "__main__":
    main()
