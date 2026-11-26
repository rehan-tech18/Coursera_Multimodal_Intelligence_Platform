
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "reviews.db"


def initialize_review_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS educator_reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id TEXT NOT NULL,
                question TEXT NOT NULL,
                ai_answer TEXT NOT NULL,
                evidence_ids TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer_notes TEXT,
                reviewed_at TEXT NOT NULL
            )
        """)


def save_review(topic_id, question, answer,
                sources, decision, notes):

    initialize_review_database()

    evidence_ids = [
        source["source_id"]
        for source in sources
    ]

    reviewed_at = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO educator_reviews
            (
                topic_id,
                question,
                ai_answer,
                evidence_ids,
                decision,
                reviewer_notes,
                reviewed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                question,
                answer,
                json.dumps(evidence_ids),
                decision,
                notes,
                reviewed_at
            )
        )

        return cursor.lastrowid


def get_review_history():
    initialize_review_database()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT *
            FROM educator_reviews
            ORDER BY review_id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]
