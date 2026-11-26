
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Load our embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "vector_database")
)

collection = client.get_collection(
    name="coursera_evidence"
)

# 3. Ask a question about the teaching content
question = "How did the instructor explain while loops?"

embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# 4. Retrieve only while-loop transcript evidence
results = collection.query(
    query_embeddings=[embedding],
    n_results=3,
    where={
        "$and": [
            {"modality": "video_transcript"},
            {"topic_id": "while_loop"}
        ]
    },
    include=["documents", "metadatas"]
)

# 5. Display the results
print("\nVIDEO TRANSCRIPT SEARCH RESULTS")

for document, metadata in zip(
    results["documents"][0],
    results["metadatas"][0]
):
    print("\nTopic:", metadata["topic_name"])
    print("Start:", metadata["start_seconds"], "seconds")
    print("End:", metadata["end_seconds"], "seconds")
    print("Transcript:", document)
    print("Source:", metadata["source_location"])
