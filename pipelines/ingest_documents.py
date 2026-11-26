
import json
import hashlib
from pathlib import Path

import pymupdf
from pptx import Presentation


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "raw" / "pdfs"
SLIDES_DIR = BASE_DIR / "data" / "raw" / "slides"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "documents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "document_chunks.json"


# ==========================================
# 2. CREATE A DOCUMENT RECORD
# ==========================================

def make_record(file_path, source_type, location, text):

    relative_path = file_path.relative_to(BASE_DIR).as_posix()

    record_id = hashlib.sha256(
        f"{relative_path}:{source_type}:{location}".encode("utf-8")
    ).hexdigest()[:20]

    return {
        "record_id": record_id,
        "course_id": "PY101",
        "source_type": source_type,
        "source_file": relative_path,
        "location": location,
        "text": text.strip(),
        "data_provenance": "user_supplied_document",
        "extraction_method": "native_text"
    }


# ==========================================
# 3. EXTRACT PDF CONTENT
# ==========================================

def process_pdf(file_path):

    records = []

    with pymupdf.open(file_path) as document:

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            if not text:
                print(
                    f"  Page {page_number}: no extractable text"
                )
                continue

            records.append(
                make_record(
                    file_path=file_path,
                    source_type="pdf",
                    location=f"page_{page_number}",
                    text=text
                )
            )

    return records


# ==========================================
# 4. EXTRACT POWERPOINT CONTENT
# ==========================================

def process_powerpoint(file_path):

    records = []
    presentation = Presentation(str(file_path))

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):

        text_parts = []

        for shape in slide.shapes:

            if shape.has_text_frame:
                text = shape.text.strip()

                if text:
                    text_parts.append(text)

            if shape.has_table:

                for row in shape.table.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    text_parts.append(" | ".join(cells))

        slide_text = "\n".join(text_parts).strip()

        if not slide_text:
            print(
                f"  Slide {slide_number}: no extractable text"
            )
            continue

        records.append(
            make_record(
                file_path=file_path,
                source_type="powerpoint",
                location=f"slide_{slide_number}",
                text=slide_text
            )
        )

    return records


# ==========================================
# 5. PROCESS ALL DOCUMENTS
# ==========================================

def main():

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.rglob("*.pdf"))
    ppt_files = sorted(SLIDES_DIR.rglob("*.pptx"))

    print("\nDOCUMENT INGESTION")
    print("=" * 50)

    print("PDF files:", len(pdf_files))
    print("PowerPoint files:", len(ppt_files))

    records = []
    failed_files = []

    for file_path in pdf_files:

        print("\nProcessing PDF:", file_path.name)

        try:
            records.extend(process_pdf(file_path))

        except Exception as error:
            print("Error:", error)
            failed_files.append(str(file_path))

    for file_path in ppt_files:

        print("\nProcessing PowerPoint:", file_path.name)

        try:
            records.extend(process_powerpoint(file_path))

        except Exception as error:
            print("Error:", error)
            failed_files.append(str(file_path))

    # Do not overwrite previously extracted data
    # if all current input folders are empty.
    if not pdf_files and not ppt_files:
        print("\nNo PDFs or PowerPoint files found.")
        print("Add documents to data/raw/pdfs or data/raw/slides.")
        print("No output file was changed.")
        return

    if not records:
        print("\nNo text was extracted.")
        print("Check whether the files contain selectable text.")
        print("No output file was changed.")
        return

    # Preserve the existing output if any source failed.
    if failed_files:
        print("\nSome files failed to process:")
        for file_name in failed_files:
            print("-", file_name)

        print("Existing output was not overwritten.")
        return

    OUTPUT_PATH.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nINGESTION COMPLETE")
    print("Extracted records:", len(records))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
