
# Coursera Multimodal Intelligence Platform

## Live Demo

[Open Coursera Multimodal Intelligence Platform](https://courseramultimodalintelligenceplatform-tukfbfojmfkqpxjjq5xcp4.streamlit.app/)


## Project Demonstration Video

[Watch the Project Demo](https://drive.google.com/file/d/1LvyYMzRYp2s_HzOwiVqHWmexpEAaCQFl/view?usp=sharing)

## TEAM CONTRIBUTION 
1	Chetna	: ai/, backend/	
2	Patrali	: data/, backend/	
3	Anshu	: pipelines/, data/	
4	Kalpesh	: vector_database/
5	Rehan	: frontend/	
6	Apoorva	: tests/	






An AI-powered educational analytics platform that helps educators
identify student learning difficulties, investigate supporting
course evidence, and review AI-generated teaching recommendations.

Built with Python, Streamlit, Google Gemini, ChromaDB, SQL,
and Retrieval-Augmented Generation (RAG).

## 1. Project Overview

Online learning platforms generate educational data from several
sources:

- Quiz attempts and scores
- Student comments and feedback
- Course videos and transcripts
- PDF lesson materials
- Video screenshots

Reviewing these sources separately can make it difficult for
educators to identify learning difficulties and decide where
additional teaching support may be useful.

The Coursera Multimodal Intelligence Platform combines these
sources into one educator-facing application.

Educators can:

1. Explore quiz performance and topic-level difficulties.
2. Ask questions about student learning challenges.
3. Retrieve relevant educational evidence.
4. Generate AI-assisted teaching recommendations.
5. Review citations and source limitations.
6. Approve, reject, or request changes to generated reports.

The application is designed as an educational prototype.

## 2. Technology Stack

| Component | Technology |
| --- | --- |
| Programming language | Python |
| Frontend | Streamlit |
| Data processing | Pandas |
| Video processing | OpenCV |
| Speech transcription | Faster-Whisper |
| LLM and visual analysis | Google Gemini |
| Vector database | ChromaDB |
| Text embeddings | Sentence Transformers |
| PDF extraction | PyMuPDF |
| PowerPoint extraction | python-pptx |
| Review storage | SQLite |
| Retrieval architecture | Multimodal RAG |

The application uses Streamlit to interact directly with
Python backend functions. No separate FastAPI server is required.

## 3. Main Features

### Learning Analytics Dashboard

Displays topic-level learning difficulties using quiz results
and student feedback.

The dashboard helps educators examine where learners may need
additional instructional support.

### Multimodal Evidence Retrieval

Retrieves educational information from multiple modalities:

- Quiz analytics
- Learner comments
- Video transcript segments
- Extracted course documents
- Video frame descriptions

Evidence is indexed in a persistent ChromaDB collection.

### AI Educator Assistant

Uses Gemini and retrieved course evidence to produce:

- Key findings
- Student difficulty analysis
- Supporting evidence
- Suggested teaching improvements
- Limitations and uncertainties

### Video Frame Analysis

Selected video frames are processed using Gemini Vision.

The resulting descriptions are indexed and associated with
candidate video timestamps and approximate topic ranges.

Timestamp and topic assignments are provisional and should
not be interpreted as manually verified ground truth.

### Citation Validation

Generated reports contain evidence references.

The application checks whether cited evidence IDs correspond
to retrieved records.

This validation detects missing or invalid citation IDs.

It does not independently prove that every cited claim is
fully supported by its source.

### Educator Review and Approval

Educators can review AI-generated insights and choose:

- Approve
- Request changes
- Reject

The application stores review decisions, comments, answers,
evidence identifiers, and timestamps in SQLite.

## 4. System Architecture

Educational Data Sources
        |
        v
Data Ingestion and Processing
        |
        +-- Quiz Analytics
        +-- Student Comments
        +-- Video Transcription
        +-- Video Frame Extraction
        +-- Gemini Visual Analysis
        +-- PDF / PPTX Extraction
        |
        v
Text Embeddings and ChromaDB
        |
        v
Topic-Aware Evidence Retrieval
        |
        v
Gemini AI Report Generation
        |
        v
Citation-ID Validation
        |
        v
Streamlit Educator Dashboard
        |
        v
Educator Review and Approval
        |
        v
SQLite Database

## 5. Project Structure

```text
coursera-intelligence-platform/
|
|-- ai/
|   |-- rag_pipeline.py
|   |-- citation_validator.py
|   |-- index_documents.py
|   |-- index_visual_analysis.py
|   |-- align_visual_evidence.py
|
|-- backend/
|   |-- review_store.py
|
|-- frontend/
|   |-- app.py
|   |-- pages/
|       |-- 2_Unified_Educator_Assistant.py
|
|-- pipelines/
|   |-- ingest_documents.py
|   |-- analyze_video_frames.py
|   |-- analyze_while_loop_frames.py
|
|-- tests/
|   |-- evaluate_rag.py
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- vector_database/
|
|-- requirements.txt
|-- .env
|-- .gitignore
|-- README.md
```

The diagram lists the primary files. Additional ingestion,
transcription, analytics, and indexing scripts may also be
present in the project.

## 6. Example Use Case

An educator wants to investigate the question:

"Why are students struggling with Python while loops?"

The system:

1. Retrieves while-loop quiz performance.
2. Retrieves student comments about while loops.
3. Searches relevant course transcripts.
4. Searches available lesson documents.
5. Retrieves eligible visual descriptions from the video.
6. Generates an evidence-based AI report.
7. Validates evidence reference identifiers.
8. Presents the report for educator review.

In the current demonstration dataset, while loops have:

- 120 quiz attempts
- 80 incorrect attempts
- 66.67% incorrect-answer rate

These are figures from the demonstration dataset, not
independently verified Coursera production analytics.

## 7. Installation

### Requirements

- Python
- pip
- A Google Gemini API key
- Access to the project source code
- Required input datasets

### Create a virtual environment

From the project root:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Some dependencies may require a Python version compatible
with their published wheels.

Use a supported version of Python if installation fails.

### Configure the Gemini API key

Create a `.env` file in the project root.

Add:

```env
GEMINI_API_KEY=your_api_key_here
```

Never publish your actual API key.

## 8. Data Preparation

The application requires processed educational evidence
and a populated ChromaDB collection.

The project includes scripts for:

- Processing quiz and feedback data
- Transcribing video content
- Extracting video frames
- Extracting PDF and PowerPoint text
- Building vector embeddings
- Indexing multimodal evidence

The current local prototype uses a sample Python tutorial
video and demonstration educational datasets.

The original media files and generated data are not
necessarily included in the public repository.

To reproduce the complete dataset, the required source
files must be supplied and the ingestion pipelines rerun.

## 9. Running the Application

From the project root, execute:

```powershell
python -m streamlit run frontend/app.py
```

Open:

http://localhost:8501

Use the dashboard to explore course analytics.

Open the Unified Educator Assistant page to generate
AI-assisted learning insights and manage educator reviews.

## 10. Evaluation

The project contains a structural RAG evaluation script:

```powershell
python tests/evaluate_rag.py
```

The current local evaluation produced:

- Topics tested: 5
- Topics passed: 5
- Topics failed: 0
- Invalid citation detection: PASS

The evaluated topics were:

1. While loops
2. For loops
3. Functions
4. Conditionals
5. Error handling

The evaluation checks structural properties such as
topic consistency, duplicate evidence, source completeness,
and citation-ID validity.

Passing these checks does not establish factual correctness,
semantic relevance, or claim-level citation grounding.

## 11. Current Prototype Status

The locally tested prototype includes:

- Quiz and learning difficulty analytics
- Multimodal evidence indexing
- Topic-aware retrieval
- Gemini-generated educational reports
- Evidence-ID validation
- Streamlit dashboard and educator interface
- SQLite educator review storage
- Automated structural RAG tests

Additional improvements are needed before production use.

## 12. Limitations

### Demonstration Data

Several inputs are synthetic or independently authored
demonstration materials.

Results must not be presented as actual Coursera
student outcomes.

### Topic Alignment

Video topic ranges are approximate and can overlap.

Visual descriptions may inherit provisional topic labels.

### Citation Grounding

The citation validator checks reference IDs.

It does not fully verify claim-level factual support.

### Model Reliability

Gemini-generated descriptions and recommendations
may contain errors or unsupported inferences.

Human review remains necessary.

### Deployment

The current implementation is a locally tested
Streamlit prototype.

Production authentication, access controls,
monitoring, privacy safeguards, and scalability
have not been fully implemented.

## 13. Future Improvements

Potential improvements include:

- Human-reviewed answer evaluation
- Stronger claim-to-source grounding checks
- More precise video topic segmentation
- Real course datasets with appropriate permissions
- Multiple-course support
- User authentication and role-based permissions
- Deployment and monitoring
- Automated ingestion workflows

## 14. Intended Users

The primary intended users are educators, instructors,
course administrators, and instructional support teams.

The platform supports educators' decision-making rather
than automatically replacing human teaching judgment.

## 15. Project Purpose

This project demonstrates practical integration of:

- Data analytics
- Python data engineering
- Natural language processing
- Multimodal AI
- Retrieval-Augmented Generation
- Vector databases
- LLM API integration
- AI-assisted educational decision support

It is intended as a learning and portfolio project.
