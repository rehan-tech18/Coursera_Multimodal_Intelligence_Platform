
import json
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "video_analysis"
    / "video_frame_analysis.json"
)

DATABASE_PATH = BASE_DIR / "vector_database"
COLLECTION_NAME = "coursera_evidence"

MODALITY = "video_visual_analysis"
ID_PREFIX = "visual_analysis_"


# ==================================================
# 2. LOAD GEMINI FRAME DESCRIPTIONS
# ==================================================

def load_visual_analysis():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Video analysis JSON was not found.\n"
            "Run this command first:\n"
            "python pipelines/analyze_video_frames.py"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Video analysis JSON must contain a list."
        )

    return records


# ==================================================
# 3. PREPARE CHROMADB RECORDS
# ==================================================

def prepare_records(records):

    ids = []
    documents = []
    metadatas = []

    seen_sources = set()

    for record in records:

        if record.get("analysis_status") != "completed":
            continue

        description = str(
            record.get("description") or ""
        ).strip()

        source_file = str(
            record.get("source_file") or ""
        ).strip()

        if not description or not source_file:
            continue

        if source_file in seen_sources:
            print(
                "Skipping duplicate frame:",
                source_file
            )
            continue

        seen_sources.add(source_file)

        digest = hashlib.sha256(
            source_file.encode("utf-8")
        ).hexdigest()[:24]

        record_id = f"{ID_PREFIX}{digest}"

        metadata = {
            "modality": MODALITY,
            "course_id": str(
                record.get("course_id", "PY101")
            ),
            "video_id": str(
                record.get("video_id", "PY101_V01")
            ),
            "topic_id": "unassigned",
            "source_id": record_id,
            "source_file": source_file,
            "source_location": source_file,
            "source_type": "video_frame",
            "analysis_model": str(
                record.get("model", "unknown")
            ),
            "provenance": "gemini_visual_analysis"
        }

        # Only save a timestamp if it was actually
        # established by the source processing.
        timestamp = record.get("timestamp_seconds")

        if timestamp is not None:
            metadata["start_seconds"] = float(timestamp)

        ids.append(record_id)
        documents.append(description)
        metadatas.append(metadata)

    return ids, documents, metadatas


# ==================================================
# 4. INDEX DESCRIPTIONS INTO CHROMADB
# ==================================================

def index_visual_evidence():

    print("\nVIDEO VISUAL EVIDENCE INDEXING")
    print("=" * 55)

    records = load_visual_analysis()

    ids, documents, metadatas = prepare_records(
        records
    )

    print("Visual analysis records found:", len(records))
    print("Valid descriptions:", len(ids))

    if not ids:
        print(
            "\nNo completed descriptions are available."
        )
        print(
            "ChromaDB has not been changed."
        )
        return

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Generating visual-description embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    before_count = collection.count()

    # Reuse stable IDs when rerunning the script.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # Delete only obsolete records that were created
    # by this visual-analysis pipeline.
    existing = collection.get(
        where={"modality": MODALITY},
        include=[]
    )

    current_ids = set(ids)

    stale_ids = [
        record_id
        for record_id in existing["ids"]
        if record_id.startswith(ID_PREFIX)
        and record_id not in current_ids
    ]

    if stale_ids:
        collection.delete(ids=stale_ids)

    after_count = collection.count()

    print("\nINDEXING COMPLETE")
    print("=" * 55)
    print("Database records before:", before_count)
    print("Database records after:", after_count)
    print("Visual descriptions indexed:", len(ids))
    print("Old descriptions removed:", len(stale_ids))

    # ==============================================
    # 5. TEST SEARCHING VISUAL DESCRIPTIONS
    # ==============================================

    question = (
        "What Python code or programming concepts "
        "are visible in the teaching video?"
    )

    query_embedding = model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, len(ids)),
        where={"modality": MODALITY},
        include=["documents", "metadatas", "distances"]
    )

    print("\nVISUAL EVIDENCE SEARCH TEST")
    print("=" * 55)
    print("Question:", question)

    for index, description in enumerate(
        results["documents"][0],
        start=1
    ):

        metadata = results["metadatas"][0][index - 1]

        print(f"\nRESULT {index}")
        print("Frame:", metadata["source_file"])
        print("Description:", description[:500])

        if "start_seconds" in metadata:
            print(
                "Timestamp:",
                metadata["start_seconds"]
            )
        else:
            print("Timestamp: not yet mapped")

    print("\nVISUAL EVIDENCE TEST FINISHED")


# ==================================================
# 6. RUN
# ==================================================

if __name__ == "__main__":
    index_visual_evidence()
