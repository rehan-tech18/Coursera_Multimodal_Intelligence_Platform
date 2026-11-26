
from pathlib import Path
from collections import Counter
import re
import sys

import chromadb


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATABASE_PATH = BASE_DIR / "vector_database"
COLLECTION_NAME = "coursera_evidence"


# ==========================================
# 2. NORMALIZE TEXT FOR DUPLICATE CHECKING
# ==========================================

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ==========================================
# 3. CHECK CHROMADB RECORDS
# ==========================================

def audit_database():

    print("\n1. CHROMADB DATABASE AUDIT")
    print("=" * 55)

    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    result = collection.get(
        include=["documents", "metadatas"]
    )

    ids = result["ids"]
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []

    print("Total records:", len(ids))

    modality_counts = Counter(
        metadata.get("modality", "unknown")
        for metadata in metadatas
        if isinstance(metadata, dict)
    )

    print("\nRecords by evidence type:")

    for modality, count in sorted(
        modality_counts.items()
    ):
        print(f"  {modality}: {count}")

    # Check for exact or whitespace-only duplicates.
    normalized = [
        normalize_text(document)
        for document in documents
        if document and document.strip()
    ]

    frequencies = Counter(normalized)

    duplicate_groups = {
        text: count
        for text, count in frequencies.items()
        if count > 1
    }

    print("\nRepeated evidence text:")

    if duplicate_groups:
        for text, count in sorted(
            duplicate_groups.items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]:
            print(f"  Repeated {count} times: {text[:120]}")
    else:
        print("  No exact text duplicates detected.")

    # Check required provenance information.
    missing_source = []

    for index, metadata in enumerate(metadatas):

        metadata = metadata or {}

        if not metadata.get("source_id"):
            missing_source.append(ids[index])

    print("\nRecords missing source_id:",
          len(missing_source))

    if missing_source:
        print("Examples:", missing_source[:5])

    # Check the document index.
    document_metadata = [
        metadata
        for metadata in metadatas
        if metadata
        and metadata.get("modality") == "course_document"
    ]

    print("\nPDF / PowerPoint evidence:")

    if document_metadata:

        files = sorted({
            metadata.get("source_file", "unknown")
            for metadata in document_metadata
        })

        print("Indexed document chunks:",
              len(document_metadata))

        for file_name in files:
            print("  Source:", file_name)

    else:
        print(
            "  No course_document records found."
        )
        print(
            "  Run python ai/index_documents.py "
            "after document ingestion."
        )

    return collection


# ==========================================
# 4. CHECK ACTUAL RAG RETRIEVAL
# ==========================================

def audit_retrieval():

    print("\n2. RAG RETRIEVAL AUDIT")
    print("=" * 55)

    from ai.rag_pipeline import retrieve_evidence

    topic_id = "while_loop"

    question = (
        "Why do students struggle with Python "
        "while loops and infinite loops?"
    )

    evidence = retrieve_evidence(
        question=question,
        topic_id=topic_id,
        top_k=3
    )

    print("Topic:", topic_id)
    print("Question:", question)
    print("Retrieved evidence:", len(evidence))

    references = []
    normalized_texts = []
    modalities = Counter()
    missing_fields = []

    for item in evidence:

        reference = str(
            item.get("reference", "")
        )

        modality = str(
            item.get("modality", "unknown")
        )

        source_id = item.get("source_id")
        text = str(item.get("text", ""))

        references.append(reference)
        modalities[modality] += 1
        normalized_texts.append(
            normalize_text(text)
        )

        if not reference or not source_id or not text:
            missing_fields.append(reference)

        print("\nReference:", reference)
        print("Modality:", modality)
        print("Source ID:", source_id)
        print(
            "Location:",
            item.get("source_location")
        )
        print("Text preview:", text[:180])

    print("\nEvidence type breakdown:")

    for modality, count in sorted(
        modalities.items()
    ):
        print(f"  {modality}: {count}")

    unique_references = (
        len(references) == len(set(references))
    )

    duplicate_counts = Counter(
        normalized_texts
    )

    repeated_results = sum(
        count - 1
        for count in duplicate_counts.values()
        if count > 1
    )

    print("\n3. RETRIEVAL CHECKS")
    print("=" * 55)

    print(
        "Unique reference labels:",
        "PASS" if unique_references else "FAIL"
    )

    print(
        "Essential evidence fields:",
        "PASS" if not missing_fields else "FAIL"
    )

    print(
        "Repeated retrieved text:",
        repeated_results
    )

    if repeated_results:
        print(
            "WARNING: Retrieved evidence includes "
            "duplicate text."
        )

    if "course_document" not in modalities:
        print(
            "NOTE: The existing RAG retrieval "
            "does not include PDF evidence."
        )

    return evidence


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

def main():

    print("\nCOURSERA EVIDENCE QUALITY AUDIT")
    print("=" * 55)

    audit_database()
    audit_retrieval()

    print("\n" + "=" * 55)
    print("AUDIT FINISHED")
    print(
        "This audit checks evidence structure, "
        "duplication and retrieval."
    )
    print(
        "It does not verify the truth of every "
        "Gemini-generated claim."
    )


if __name__ == "__main__":
    main()
