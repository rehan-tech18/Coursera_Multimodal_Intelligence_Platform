
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai.rag_pipeline import (
    retrieve_evidence,
    generate_answer
)

# 1. Define the test question
question = (
    "Why are students struggling with Python while loops? "
    "Use quiz statistics, student comments and video evidence."
)

topic_id = "while_loop"

print("\nSTEP 1: TESTING CHROMADB RETRIEVAL")
print("=" * 55)

# 2. Retrieve relevant evidence
evidence = retrieve_evidence(
    question,
    topic_id=topic_id
)

# 3. Display retrieved sources
for source in evidence:
    print(
        source["reference"],
        "|",
        source["modality"],
        "|",
        source["source_id"]
    )

    if source.get("start_seconds") is not None:
        print(
            "Video timestamp:",
            source["start_seconds"],
            "seconds"
        )

    print("Text:", source["text"][:200])
    print("-" * 55)

# 4. Validate the evidence types
modalities = {
    source["modality"]
    for source in evidence
}

required = {
    "quiz_analytics",
    "learner_comment",
    "video_transcript"
}

missing = required - modalities

if missing:
    raise AssertionError(
        f"Missing evidence types: {missing}"
    )

# 5. Check that the correct topic was retrieved
assert all(
    source["topic_id"] == topic_id
    for source in evidence
), "Incorrect topic found in evidence"

# 6. Check the known quiz statistics
analytics = [
    source for source in evidence
    if source["modality"] == "quiz_analytics"
]

assert len(analytics) == 1, (
    "Expected exactly one quiz analytics record"
)

assert "66.67" in analytics[0]["text"], (
    "Expected while-loop error rate not found"
)

# 7. Verify transcript timestamps
transcripts = [
    source for source in evidence
    if source["modality"] == "video_transcript"
]

assert all(
    source.get("start_seconds") is not None
    for source in transcripts
), "Transcript timestamp missing"

print("\nCHROMADB RETRIEVAL: PASSED")

# 8. Test Gemini generation
print("\nSTEP 2: TESTING GEMINI")
print("=" * 55)

try:
    answer = generate_answer(question, evidence)

    print("\nGEMINI ANSWER:\n")
    print(answer)

    print("\nGEMINI GENERATION: SUCCESS")

except Exception as error:
    print("\nGEMINI GENERATION FAILED")
    print("Error type:", type(error).__name__)
    print("Error message:", str(error))
    raise
