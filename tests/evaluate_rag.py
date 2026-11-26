
import json
import sys
from pathlib import Path
from collections import Counter

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai.rag_pipeline import retrieve_evidence
from ai.citation_validator import validate_answer_citations


OUTPUT_DIR = BASE_DIR / "data" / "processed" / "evaluation"
OUTPUT_FILE = OUTPUT_DIR / "rag_evaluation.json"


# ==================================================
# 2. TEST QUESTIONS
# ==================================================

TEST_CASES = [
    {
        "topic_id": "while_loop",
        "question": (
            "Why do students struggle with while "
            "loops and infinite loops? What code "
            "is visible on screen?"
        )
    },
    {
        "topic_id": "for_loop",
        "question": (
            "Why do students struggle with for "
            "loops and range()? What code is "
            "shown on screen?"
        )
    },
    {
        "topic_id": "functions",
        "question": (
            "What difficulties do students report "
            "when learning Python functions?"
        )
    },
    {
        "topic_id": "conditionals",
        "question": (
            "What difficulties do students report "
            "with Python if and else conditions?"
        )
    },
    {
        "topic_id": "error_handling",
        "question": (
            "What difficulties do students report "
            "with try except and invalid input?"
        )
    }
]


# ==================================================
# 3. NORMALIZE EVIDENCE TEXT
# ==================================================

def normalize_text(text):

    return " ".join(
        str(text).lower().split()
    )


# ==================================================
# 4. CHECK ONE TOPIC
# ==================================================

def evaluate_topic(topic_id, question):

    print("\n" + "=" * 60)
    print("TOPIC:", topic_id)
    print("QUESTION:", question)

    evidence = retrieve_evidence(
        question=question,
        topic_id=topic_id,
        top_k=3
    )

    modalities = Counter()

    problems = []
    warnings = []

    references = []
    texts = []

    topic_specific_modalities = {
        "quiz_analytics",
        "learner_comment",
        "video_transcript",
        "video_visual_analysis"
    }

    for item in evidence:

        modality = item.get("modality", "unknown")
        source_topic = item.get("topic_id")
        reference = item.get("reference", "")
        source_id = item.get("source_id", "")
        text = str(item.get("text", ""))

        modalities[modality] += 1
        references.append(reference)
        texts.append(normalize_text(text))

        # Check that evidence contains required fields.
        if not reference or not source_id or not text.strip():
            problems.append(
                f"Incomplete evidence: {reference}"
            )

        # Check topic-specific sources.
        if modality in topic_specific_modalities:

            if source_topic != topic_id:

                problems.append(
                    f"{reference}: topic mismatch "
                    f"({source_topic})"
                )

        # Documents currently have no verified
        # topic labels, so evaluate them separately.
        if modality == "course_document":

            if source_topic != topic_id:

                warnings.append(
                    f"{reference}: document topic "
                    "relevance needs review"
                )

        # Visual descriptions use approximate
        # timestamp-based topic assignments.
        if modality == "video_visual_analysis":

            if item.get(
                "topic_mapping_status"
            ) != "unique_range_match":

                problems.append(
                    f"{reference}: visual topic "
                    "mapping is not unique"
                )

            if item.get("start_seconds") is None:

                problems.append(
                    f"{reference}: visual timestamp missing"
                )

    # Check reference uniqueness.
    if len(references) != len(set(references)):

        problems.append(
            "Duplicate evidence reference IDs"
        )

    # Check exact text duplication.
    nonempty_texts = [
        text for text in texts if text
    ]

    repeated_count = (
        len(nonempty_texts)
        - len(set(nonempty_texts))
    )

    if repeated_count > 0:

        problems.append(
            f"{repeated_count} repeated evidence passages"
        )

    # Confirm that quiz analytics were retrieved.
    if modalities["quiz_analytics"] == 0:

        problems.append(
            "Quiz analytics evidence missing"
        )

    if not evidence:

        problems.append(
            "No evidence retrieved"
        )

    # Offline citation check using the actual IDs.
    if evidence:

        valid_reference = evidence[0]["reference"]

        sample_answer = (
            f"This answer cites [{valid_reference}]."
        )

        citation_result = validate_answer_citations(
            sample_answer,
            evidence
        )

        citation_passed = citation_result["passed"]

    else:

        citation_passed = False

    if not citation_passed:

        problems.append(
            "Citation validator did not accept "
            "a known evidence ID"
        )

    # Summary
    passed = len(problems) == 0

    print("\nRetrieved records:", len(evidence))

    print("\nEvidence breakdown:")

    for modality, count in sorted(
        modalities.items()
    ):
        print(f"  {modality}: {count}")

    print("\nRepeated passages:", repeated_count)

    print(
        "Structural checks:",
        "PASS" if passed else "FAIL"
    )

    if problems:

        print("\nPROBLEMS:")

        for problem in problems:
            print(" -", problem)

    if warnings:

        print("\nREVIEW WARNINGS:")

        for warning in warnings:
            print(" -", warning)

    return {
        "topic_id": topic_id,
        "question": question,
        "passed": passed,
        "evidence_count": len(evidence),
        "modalities": dict(modalities),
        "repeated_passages": repeated_count,
        "citation_validator_passed": citation_passed,
        "problems": problems,
        "warnings": warnings,
        "sources": [
            {
                "reference": item.get("reference"),
                "source_id": item.get("source_id"),
                "modality": item.get("modality"),
                "topic_id": item.get("topic_id"),
                "source_location": item.get(
                    "source_location"
                )
            }
            for item in evidence
        ]
    }


# ==================================================
# 5. TEST INVALID CITATIONS
# ==================================================

def test_invalid_citation_detection():

    evidence = [
        {
            "reference": "E1",
            "source_id": "test_record",
            "text": "Example evidence"
        }
    ]

    invalid_answer = (
        "This claim cites a nonexistent source [E999]."
    )

    result = validate_answer_citations(
        invalid_answer,
        evidence
    )

    return (
        not result["passed"]
        and "[E999]" in result["invalid_citations"]
    )


# ==================================================
# 6. RUN THE FULL EVALUATION
# ==================================================

def main():

    print("\nCOURSERA RAG QUALITY EVALUATION")
    print("=" * 60)

    results = []

    for test_case in TEST_CASES:

        try:

            result = evaluate_topic(
                topic_id=test_case["topic_id"],
                question=test_case["question"]
            )

            results.append(result)

        except Exception as error:

            print(
                "\nEVALUATION ERROR:",
                error
            )

            results.append({
                "topic_id": test_case["topic_id"],
                "question": test_case["question"],
                "passed": False,
                "error": str(error)
            })

    invalid_citation_test = (
        test_invalid_citation_detection()
    )

    passed_topics = sum(
        result["passed"]
        for result in results
    )

    total_topics = len(results)

    summary = {
        "total_topics": total_topics,
        "passed_topics": passed_topics,
        "failed_topics": total_topics - passed_topics,
        "invalid_citation_detection_passed": (
            invalid_citation_test
        )
    }

    # Save results to a JSON report.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "summary": summary,
        "results": results
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    print("Topics tested:", total_topics)
    print("Topics passed:", passed_topics)
    print(
        "Topics failed:",
        total_topics - passed_topics
    )

    print(
        "Invalid citation detection:",
        "PASS" if invalid_citation_test else "FAIL"
    )

    print("\nEvaluation report saved to:")
    print(OUTPUT_FILE)

    if (
        passed_topics != total_topics
        or not invalid_citation_test
    ):

        print("\nOVERALL RESULT: NEEDS REVIEW")
        sys.exit(1)

    print("\nOVERALL RESULT: STRUCTURAL TESTS PASS")

    print(
        "Important: Structural success does not "
        "prove semantic relevance, factual "
        "grounding or AI-answer correctness."
    )


# ==================================================
# 7. START
# ==================================================

if __name__ == "__main__":
    main()
