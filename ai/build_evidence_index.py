
import json
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Define project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FRAMES_DIR = PROCESSED_DIR / "video_frames"
DB_DIR = BASE_DIR / "vector_database"

# 2. Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Connect to persistent ChromaDB
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="coursera_evidence",
    metadata={"hnsw:space": "cosine"}
)

# 4. Prepare lists of evidence
ids = []
documents = []
metadatas = []

def add_evidence(evidence_id, text, metadata):
    if not isinstance(text, str) or not text.strip():
        return

    ids.append(str(evidence_id))
    documents.append(text.strip())
    metadatas.append(metadata)

# 5. Add student comments
comments = pd.read_csv(RAW_DIR / "student_comments.csv")

for _, row in comments.iterrows():
    add_evidence(
        evidence_id=f"comment_{row['comment_id']}",
        text=row["comment_text"],
        metadata={
            "modality": "learner_comment",
            "source_id": str(row["comment_id"]),
            "course_id": str(row["course_id"]),
            "video_id": str(row["video_id"]),
            "topic_id": str(row["topic_id"]),
            "source_location": "student_comments.csv"
        }
    )

# 6. Add verified quiz analytics
learning = pd.read_csv(
    PROCESSED_DIR / "learning_difficulty_summary.csv"
)

for _, row in learning.iterrows():
    text = (
        f"Topic: {row['topic_name']}. "
        f"Quiz attempts: {row['total_attempts']}. "
        f"Incorrect answers: {row['incorrect_answers']}. "
        f"Incorrect rate: {row['incorrect_rate_pct']} percent. "
        f"Student comment count: {row['comment_count']}."
    )

    add_evidence(
        evidence_id=f"analytics_{row['course_id']}_{row['video_id']}_{row['topic_id']}",
        text=text,
        metadata={
            "modality": "quiz_analytics",
            "source_id": str(row["topic_id"]),
            "course_id": str(row["course_id"]),
            "video_id": str(row["video_id"]),
            "topic_id": str(row["topic_id"]),
            "source_location": "learning_difficulty_summary.csv"
        }
    )

# 7. Add video-frame descriptions, if available
descriptions_path = FRAMES_DIR / "frame_descriptions.json"

if descriptions_path.exists():
    frame_descriptions = json.loads(
        descriptions_path.read_text(encoding="utf-8")
    )

    for frame in frame_descriptions:

        if frame.get("status") != "success":
            continue

        description = frame.get("description")

        if not description:
            continue

        add_evidence(
            evidence_id=f"frame_{frame['video_id']}_{frame['timestamp_seconds']}",
            text=description,
            metadata={
                "modality": "video_frame",
                "source_id": str(frame["frame_file"]),
                "course_id": "PY101",
                "video_id": str(frame["video_id"]),
                "topic_id": ",".join(frame.get("topic_names", [])),
                "timestamp_seconds": int(frame["timestamp_seconds"]),
                "source_location": str(frame["frame_file"]),
                "model": str(frame["model"])
            }
        )

# 8. Generate embeddings
print("Evidence records prepared:", len(documents))

if not documents:
    raise RuntimeError("No evidence available to index.")

embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
).tolist()

# 9. Store records in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Records stored in collection:", collection.count())
print("Database location:", DB_DIR)
