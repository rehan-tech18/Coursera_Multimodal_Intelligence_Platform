
import re
import sys
from pathlib import Path


# ==================================================
# 1. PROJECT SETUP
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ==================================================
# 2. EXTRACT CITATIONS FROM GEMINI ANSWERS
# ==================================================

def extract_citation_groups(answer):

    # Finds bracketed text such as:
    # [E1], [E2, E3], [Rule 7], [E99]
    return re.findall(
        r"\[([^\[\]\n]{1,100})\]",
        answer
    )


# ==================================================
# 3. VALIDATE CITATIONS
# ==================================================

def validate_answer_citations(answer, evidence):

    valid_ids = {
        str(item["reference"])
        for item in evidence
    }

    groups = extract_citation_groups(answer)

    cited_ids = []
    invalid_citations = []

    for group in groups:

        # Accept citations such as:
        # [E1]
        # [E1, E2]
        # [E1, E2, E3]
        # [E1; E2]
        valid_format = re.fullmatch(
            r"E\d+(?:\s*[,;]\s*E\d+)*",
            group.strip()
        )

        if not valid_format:

            invalid_citations.append(
                f"[{group}]"
            )

            continue

        references = re.findall(
            r"E\d+",
            group
        )

        for reference in references:

            cited_ids.append(reference)

            if reference not in valid_ids:

                invalid_citations.append(
                    f"[{reference}]"
                )

    unique_cited_ids = list(
        dict.fromkeys(cited_ids)
    )

    invalid_citations = list(
        dict.fromkeys(invalid_citations)
    )

    uncited_ids = sorted(
        valid_ids - set(unique_cited_ids),
        key=lambda value: int(value[1:])
    )

    passed = (
        bool(unique_cited_ids)
        and not invalid_citations
    )

    return {
        "passed": passed,
        "valid_evidence_ids": sorted(
            valid_ids,
            key=lambda value: int(value[1:])
        ),
        "cited_evidence_ids": unique_cited_ids,
        "invalid_citations": invalid_citations,
        "uncited_evidence_ids": uncited_ids,
        "citation_count": len(cited_ids)
    }


# ==================================================
# 4. PRINT VALIDATION RESULTS
# ==================================================

def display_validation(result):

    print("\nCITATION VALIDATION")
    print("=" * 55)

    print(
        "Available evidence:",
        len(result["valid_evidence_ids"])
    )

    print(
        "Citation occurrences:",
        result["citation_count"]
    )

    print(
        "Unique evidence IDs cited:",
        len(result["cited_evidence_ids"])
    )

    print(
        "Invalid citations:",
        len(result["invalid_citations"])
    )

    if result["invalid_citations"]:

        print("\nINVALID REFERENCES:")

        for reference in result["invalid_citations"]:
            print("  ", reference)

    if result["uncited_evidence_ids"]:

        print("\nRETRIEVED BUT NOT CITED:")

        print(
            "  ",
            ", ".join(
                result["uncited_evidence_ids"]
            )
        )

    print("\nRESULT:")

    if result["passed"]:
        print("PASS - All cited IDs exist in the evidence.")
    else:
        print("FAIL - Missing or invalid evidence citations.")


# ==================================================
# 5. OFFLINE VALIDATOR TEST
# ==================================================

def run_offline_test():

    print("\nOFFLINE CITATION TEST")
    print("=" * 55)

    sample_evidence = [
        {"reference": "E1"},
        {"reference": "E2"},
        {"reference": "E3"},
        {"reference": "E8"}
    ]

    good_answer = (
        "The quiz incorrect rate was high [E1]. "
        "Students reported difficulty [E2, E3]. "
        "The notes explain loops [E8]."
    )

    bad_answer = (
        "The quiz incorrect rate was high [E1]. "
        "The source proves improvement [E99]. "
        "Follow the instruction [Rule 7]."
    )

    good_result = validate_answer_citations(
        good_answer,
        sample_evidence
    )

    bad_result = validate_answer_citations(
        bad_answer,
        sample_evidence
    )

    assert good_result["passed"]

    assert not bad_result["passed"]

    assert "[E99]" in bad_result["invalid_citations"]

    assert "[Rule 7]" in bad_result["invalid_citations"]

    print("Valid citation test: PASS")
    print("Invalid citation detection: PASS")
    print("Rule citation detection: PASS")

    print("\nExample of rejected references:")

    display_validation(bad_result)


# ==================================================
# 6. VALIDATE A REAL GEMINI REPORT
# ==================================================

def run_live_test():

    from ai.rag_pipeline import ask_course

    print("\nGENERATING LIVE GEMINI REPORT")
    print("=" * 55)

    question = (
        "Why do students struggle with Python "
        "while loops? Analyse the quiz performance, "
        "student comments, video transcripts and "
        "PDF notes. Suggest possible teaching "
        "improvements without claiming proven results."
    )

    result = ask_course(
        question=question,
        topic_id="while_loop"
    )

    answer = result["answer"]
    evidence = result["sources"]

    print("\nGEMINI ANSWER")
    print("=" * 55)
    print(answer)

    validation = validate_answer_citations(
        answer,
        evidence
    )

    display_validation(validation)

    if not validation["passed"]:

        print(
            "\nWARNING: Do not treat this report as "
            "citation-verified until the invalid "
            "references are corrected."
        )

    return validation["passed"]


# ==================================================
# 7. MAIN
# ==================================================

def main():

    run_offline_test()

    if "--live" in sys.argv:

        passed = run_live_test()

        if not passed:
            sys.exit(1)

    else:

        print(
            "\nOffline tests completed. "
            "Use --live to test a real Gemini answer."
        )


if __name__ == "__main__":
    main()
