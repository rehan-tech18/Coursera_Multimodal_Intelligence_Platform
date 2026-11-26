
from pathlib import Path
import json

from faster_whisper import WhisperModel

# 1. Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR = RAW_DIR / "transcripts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Find the original MP4
videos = list(RAW_DIR.glob("*.mp4"))

if len(videos) != 1:
    raise ValueError(
        f"Expected exactly one video, found {len(videos)}"
    )

video_path = videos[0]

# 3. Load the speech recognition model
model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8"
)

# 4. Transcribe spoken audio
segments, info = model.transcribe(
    str(video_path),
    language="en",
    beam_size=5,
    vad_filter=True
)

# 5. Save timestamped segments
records = []

for index, segment in enumerate(segments, start=1):

    record = {
        "segment_id": f"PY101_V01_T{index:04d}",
        "video_id": "PY101_V01",
        "start_seconds": round(segment.start, 2),
        "end_seconds": round(segment.end, 2),
        "text": segment.text.strip(),
        "source_video": video_path.name,
        "transcription_model": "base.en"
    }

    records.append(record)

    print(
        f"[{segment.start:.1f}s - {segment.end:.1f}s]",
        segment.text.strip()
    )

# 6. Save results to JSON
output_path = OUTPUT_DIR / "video_transcript.json"

output_path.write_text(
    json.dumps(records, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nTranscription complete!")
print("Segments:", len(records))
print("Saved:", output_path)
