
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Locate project folders
BASE_DIR = Path(__file__).resolve().parent.parent

TRANSCRIPT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "transcripts"
    / "mapped_transcript.csv"
)

DB_DIR = BASE_DIR / "vector_database"

# 2. Check the transcript mapping exists
if not TRANSCRIPT_PATH.exists():
    raise FileNotFoundError(
        "Mapped transcript not found. "
        "Run python pipelines/map_transcript_topics.py first."
    )

# 3. Load mapped transcripts
transcripts = pd.read_csv(TRANSCRIPT_PATH)

print("Mapped transcript records:", len(transcripts))

if transcripts.empty:
    raise ValueError("The mapped transcript file is empty.")

# 4. Load the same embedding model used previously
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# 5. Connect to the existing ChromaDB collection
client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_collection(
    name="coursera_evidence"
)

# 6. Prepare transcript records
ids = []
documents = []
metadatas = []

for _, row in transcripts.iterrows():

    transcript_id = (
        f"transcript_{row['segment_id']}_{row['topic_id']}"
    )

    transcript_text = str(row["text"]).strip()

    if not transcript_text:
        continue

    ids.append(transcript_id)

    documents.append(transcript_text)

    metadatas.append({
        "modality": "video_transcript",
        "source_id": str(row["segment_id"]),
        "course_id": str(row["course_id"]),
        "video_id": str(row["video_id"]),
        "topic_id": str(row["topic_id"]),
        "topic_name": str(row["topic_name"]),
        "start_seconds": float(row["start_seconds"]),
        "end_seconds": float(row["end_seconds"]),
        "source_location": str(row["source_video"]),
        "mapping_method": str(row["mapping_method"])
    })

# 7. Convert transcript text to embeddings
embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
).tolist()

# 8. Store transcripts in ChromaDB
# Remove outdated transcript-topic associations
collection.delete(
    where={"modality": "video_transcript"}
)
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("\nTRANSCRIPT INDEXING COMPLETE")
print("Transcript records indexed:", len(ids))
print("Total collection records:", collection.count())
print("Database:", DB_DIR)
