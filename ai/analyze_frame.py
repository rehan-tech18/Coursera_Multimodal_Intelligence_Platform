
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Locate the project root
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Load API configuration
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# 3. Create the Gemini client
client = genai.Client(api_key=api_key)

# 4. Locate one extracted frame
frame_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "video_frames"
    / "PY101_V01_frame_00200.jpg"
)

if not frame_path.is_file():
    raise FileNotFoundError(f"Frame not found: {frame_path}")

# 5. Read the image as bytes
image_bytes = frame_path.read_bytes()

# 6. Prepare the instruction
prompt = """
You are analyzing a frame from a Python teaching video.

Describe:
1. What is visibly shown in this image?
2. What Python code or concepts are visible?
3. Is any program output or error message visible?
4. What might a beginner find confusing?

Rules:
- Only describe information supported by this frame.
- Do not invent dialogue or explanations from the instructor.
- If code is too small to read, say so.
- Do not claim students struggled without student feedback.

Give a concise, structured answer.
"""

# 7. Send the image and prompt to Gemini
response = client.models.generate_content(
    model=model_name,
    contents=[
        prompt,
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
    ]
)

# 8. Display the result
if not response.text:
    raise RuntimeError("Gemini returned no text description.")

print("\nVISUAL FRAME ANALYSIS")
print("=" * 50)
print("Frame:", frame_path.name)
print("Model:", model_name)
print("\nGemini description:")
print(response.text)
