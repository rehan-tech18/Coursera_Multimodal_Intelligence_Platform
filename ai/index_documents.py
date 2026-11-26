
import json
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "documents"
    / "document_chunks.json"
)

DATABASE_PATH = BASE_DIR / "vector_database"

COLLECTION_NAME = "coursera_evidence"


# ==========================================
# 2. TEXT CHUNKING
# ==========================================

def split_text(text, chunk_size=900, overlap=150):

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# ==========================================
# 3. LOAD EXTRACTED DOCUMENTS
# ==========================================

def load_documents():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Document extraction file not found. "
            "Run python pipelines/ingest_documents.py first."
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Document file must contain a list of records."
        )

    return records


# ==========================================
# 4. CREATE CHROMADB RECORDS
# ==========================================

def prepare_records(records):

    ids = []
    documents = []
    metadatas = []

    for record in records:

        text = record["text"]
        chunks = split_text(text)

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            source_file = record["source_file"]
            location = record["location"]

            unique_value = (
                f"{source_file}:{location}:{chunk_number}"
            )

            record_id = hashlib.sha256(
                unique_value.encode("utf-8")
            ).hexdigest()[:24]

            ids.append(f"course_doc_{record_id}")

            documents.append(chunk)

            metadatas.append({
                "modality": "course_document",
                "course_id": record.get(
                    "course_id", "PY101"
                ),
                "topic_id": "unassigned",
                "source_id": record_id,
                "source_type": record["source_type"],
                "source_file": source_file,
                "source_location": (
                    f"{source_file}#{location}"
                ),
                "location": location,
                "chunk_number": chunk_number,
                "provenance": record.get(
                    "data_provenance",
                    "user_supplied_document"
                )
            })

    return ids, documents, metadatas


# ==========================================
# 5. INDEX DOCUMENTS
# ==========================================

def main():

    print("\nDOCUMENT CHROMADB INDEXING")
    print("=" * 50)

    records = load_documents()

    ids, documents, metadatas = prepare_records(
        records
    )

    if not ids:
        print("No document text available to index.")
        return

    print("Document pages/slides:", len(records))
    print("Searchable chunks:", len(ids))

    # Use the same embedding model as our RAG pipeline.
    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Generating embeddings...")

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    # Connect to our existing database.
    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    before_count = collection.count()

    print("\nExisting database records:", before_count)

    # Add new document records or update existing ones.
    # Existing quiz/comment/transcript records are preserved.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # Remove old document chunks no longer present
    # in the latest successfully extracted documents.
    old_documents = collection.get(
        where={"modality": "course_document"},
        include=["metadatas"]
    )

    stale_ids = [
        old_id
        for old_id in old_documents["ids"]
        if old_id not in set(ids)
    ]

    if stale_ids:
        collection.delete(ids=stale_ids)

    after_count = collection.count()

    print("\nINDEXING COMPLETE")
    print("Records before:", before_count)
    print("Records after:", after_count)
    print("Document chunks indexed:", len(ids))
    print("Old document chunks removed:", len(stale_ids))

    # ======================================
    # 6. TEST DOCUMENT RETRIEVAL
    # ======================================

    question = (
        "How can I prevent an infinite while loop?"
    )

    print("\nTEST QUESTION:")
    print(question)

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(3, len(ids)),
        where={"modality": "course_document"},
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print("\nRETRIEVED DOCUMENT EVIDENCE")
    print("=" * 50)

    for i, document in enumerate(
        results["documents"][0],
        start=1
    ):

        metadata = results["metadatas"][0][i - 1]

        print(f"\nRESULT {i}")
        print("Source:", metadata["source_file"])
        print("Location:", metadata["location"])
        print("Text:", document[:400])

    print("\nDocument indexing and search test finished.")


if __name__ == "__main__":
    main()
