
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "vector_database")
)

collection = client.get_collection(
    name="coursera_evidence"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "Why are students struggling with Python while loops?"

query_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(5, collection.count())
)

print("\nSEARCH RESULTS")

for document, metadata in zip(
    results["documents"][0],
    results["metadatas"][0]
):
    print("\nSource:", metadata.get("source_id"))
    print("Modality:", metadata.get("modality"))
    print("Content:", document[:300])
