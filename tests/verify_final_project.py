
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import chromadb
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKS = []

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "frontend/app.py",
    "frontend/pages/2_Unified_Educator_Assistant.py",
    "ai/rag_pipeline.py",
    "ai/citation_validator.py",
    "ai/index_documents.py",
    "ai/index_visual_analysis.py",
    "ai/align_visual_evidence.py",
    "backend/review_store.py",
    "tests/evaluate_rag.py",
]

VISUAL_JSON = (
    BASE_DIR
    / "data"
    / "processed"
    / "video_analysis"
    / "video_frame_analysis.json"
)

QUIZ_SUMMARY = (
    BASE_DIR
    / "data"
    / "processed"
    / "learning_difficulty_summary.csv"
)

SQLITE_FILE = BASE_DIR / "data" / "reviews.db"

VECTOR_DIR = BASE_DIR / "vector_database"

COLLECTION_NAME = "coursera_evidence"


# ==================================================
# 2. RECORD TEST RESULT
# ==================================================

def check(name, passed, details=""):

    status = "PASS" if passed else "FAIL"

    CHECKS.append({
        "test": name,
        "passed": bool(passed),
        "details": str(details)
    })

    print(f"[{status}] {name}")

    if details:
        print(f"       {details}")


# ==================================================
# 3. VERIFY PROJECT FILES
# ==================================================

def check_project_files():

    print("\n1. PROJECT FILES")
    print("-" * 55)

    missing = [
        name
        for name in REQUIRED_FILES
        if not (BASE_DIR / name).is_file()
    ]

    check(
        "Required project files",
        len(missing) == 0,
        (
            "All required files found."
            if not missing
            else "Missing: " + ", ".join(missing)
        )
    )


# ==================================================
# 4. VERIFY QUIZ ANALYTICS
# ==================================================

def check_quiz_analytics():

    print("\n2. QUIZ ANALYTICS")
    print("-" * 55)

    if not QUIZ_SUMMARY.exists():

        check(
            "Quiz analytics",
            False,
            "Summary CSV is missing."
        )
        return

    try:

        dataframe = pd.read_csv(QUIZ_SUMMARY)

        check(
            "Quiz analytics",
            not dataframe.empty,
            f"{len(dataframe)} topic records found."
        )

    except Exception as error:

        check(
            "Quiz analytics",
            False,
            str(error)
        )


# ==================================================
# 5. VERIFY VISUAL DESCRIPTIONS
# ==================================================

def check_visual_analysis():

    print("\n3. VIDEO VISUAL ANALYSIS")
    print("-" * 55)

    if not VISUAL_JSON.exists():

        check(
            "Visual analysis JSON",
            False,
            "File does not exist."
        )
        return

    try:

        records = json.loads(
            VISUAL_JSON.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(records, list):
            raise ValueError(
                "Expected a JSON list."
            )

        completed = [
            record
            for record in records
            if record.get("analysis_status")
            == "completed"
        ]

        check(
            "Video visual descriptions",
            len(completed) > 0,
            (
                f"{len(completed)} completed analyses "
                f"out of {len(records)} records."
            )
        )

        source_files = [
            str(record.get("source_file", ""))
            for record in completed
        ]

        check(
            "Unique visual source files",
            len(source_files) == len(
                set(source_files)
            ),
            f"{len(source_files)} source entries."
        )

    except Exception as error:

        check(
            "Video visual descriptions",
            False,
            str(error)
        )


# ==================================================
# 6. VERIFY CHROMADB
# ==================================================

def check_vector_database():

    print("\n4. CHROMADB")
    print("-" * 55)

    if not VECTOR_DIR.exists():

        check(
            "ChromaDB",
            False,
            "Database directory missing."
        )
        return

    try:

        client = chromadb.PersistentClient(
            path=str(VECTOR_DIR)
        )

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        total = collection.count()

        check(
            "ChromaDB collection",
            total > 0,
            f"{total} evidence records found."
        )

        visual_records = collection.get(
            where={
                "modality": "video_visual_analysis"
            },
            include=["metadatas"]
        )

        visual_count = len(
            visual_records["ids"]
        )

        check(
            "Indexed visual evidence",
            visual_count > 0,
            f"{visual_count} visual records found."
        )

        unique_mappings = sum(
            1
            for metadata in visual_records["metadatas"]
            if metadata
            and metadata.get(
                "topic_mapping_status"
            ) == "unique_range_match"
            and metadata.get(
                "timestamp_status"
            ) == "candidate_image_match"
        )

        check(
            "Visual topic alignment",
            unique_mappings > 0,
            (
                f"{unique_mappings} visual records "
                "have unique provisional topic mappings."
            )
        )

    except Exception as error:

        check(
            "ChromaDB",
            False,
            str(error)
        )


# ==================================================
# 7. VERIFY SQLITE REVIEWS
# ==================================================

def check_sqlite():

    print("\n5. EDUCATOR REVIEW DATABASE")
    print("-" * 55)

    if not SQLITE_FILE.exists():

        check(
            "SQLite review database",
            False,
            "Database file is missing."
        )
        return

    connection = None

    try:

        # Read-only connection prevents creation
        # or modification of the database.
        connection = sqlite3.connect(
            SQLITE_FILE.as_uri() + "?mode=ro",
            uri=True
        )

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'"
            )
        }

        table_exists = (
            "educator_reviews" in tables
        )

        check(
            "Educator review table",
            table_exists,
            f"Available tables: {sorted(tables)}"
        )

        if not table_exists:
            return

        count = connection.execute(
            "SELECT COUNT(*) "
            "FROM educator_reviews"
        ).fetchone()[0]

        check(
            "Saved educator reviews",
            count > 0,
            f"{count} reviews stored."
        )

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(educator_reviews)"
            )
        }

        expected_columns = {
            "review_id",
            "topic_id",
            "question",
            "ai_answer",
            "evidence_ids",
            "decision",
            "reviewer_notes",
            "reviewed_at"
        }

        missing = expected_columns - columns

        check(
            "Review database schema",
            len(missing) == 0,
            (
                "All expected columns present."
                if not missing
                else f"Missing columns: {sorted(missing)}"
            )
        )

    except Exception as error:

        check(
            "SQLite review database",
            False,
            str(error)
        )

    finally:

        if connection is not None:
            connection.close()


# ==================================================
# 8. CHECK PRIVATE-FILE PROTECTION
# ==================================================

def check_git_safety():

    print("\n6. CONFIGURATION SAFETY")
    print("-" * 55)

    gitignore = BASE_DIR / ".gitignore"

    if not gitignore.exists():

        check(
            "Git ignore configuration",
            False,
            ".gitignore is missing."
        )
        return

    try:

        content = gitignore.read_text(
            encoding="utf-8"
        )

        active_rules = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
            and not line.lstrip().startswith("#")
        ]

        protected_items = [
            ".env",
            ".venv/",
            "vector_database/",
            "data/reviews.db"
        ]

        missing_rules = [
            item
            for item in protected_items
            if item not in active_rules
        ]

        check(
            "Git ignore configuration",
            not missing_rules,
            (
                "Expected private-file rules found."
                if not missing_rules
                else "Missing explicit rules: "
                + ", ".join(missing_rules)
            )
        )

        # If this is already a Git repository,
        # inspect whether the secret file
        # was accidentally tracked.
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                ".env"
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=False
        )

        check(
            "Git secret tracking",
            result.returncode != 0,
            (
                ".env is not tracked by Git."
                if result.returncode != 0
                else "WARNING: .env is tracked by Git!"
            )
        )

    except FileNotFoundError:

        print(
            "Git executable not available. "
            "Skipping Git tracking check."
        )

    except Exception as error:

        check(
            "Git safety inspection",
            False,
            str(error)
        )


# ==================================================
# 9. FINAL SUMMARY
# ==================================================

def main():

    print("\nCOURSERA PLATFORM FINAL VERIFICATION")
    print("=" * 55)

    print("Project:", BASE_DIR)

    check_project_files()
    check_quiz_analytics()
    check_visual_analysis()
    check_vector_database()
    check_sqlite()
    check_git_safety()

    passed = sum(
        item["passed"]
        for item in CHECKS
    )

    failed = len(CHECKS) - passed

    print("\n" + "=" * 55)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 55)

    print("Total checks:", len(CHECKS))
    print("Passed:", passed)
    print("Failed:", failed)

    if failed:

        print("\nFAILED CHECKS:")

        for item in CHECKS:

            if not item["passed"]:

                print(
                    "-",
                    item["test"],
                    ":",
                    item["details"]
                )

    report = {
        "total": len(CHECKS),
        "passed": passed,
        "failed": failed,
        "checks": CHECKS
    }

    report_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "evaluation"
        / "final_verification.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nReport saved to:", report_path)

    if failed:
        print("\nRESULT: NEEDS REVIEW")
        sys.exit(1)

    print("\nRESULT: LOCAL DATA CHECKS PASS")


if __name__ == "__main__":
    main()
