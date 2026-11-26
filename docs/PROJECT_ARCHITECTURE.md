# Project Architecture

## Project Name
Coursera Multimodal Intelligence Platform

## Overview
An AI-powered educational analytics platform that helps educators identify student learning difficulties and generate teaching recommendations.

## Main Components

1. Frontend – Streamlit interface for educators.
2. Backend – Python-based application logic.
3. AI and RAG Pipeline – Retrieves relevant educational evidence and generates AI insights.
4. Vector Database – ChromaDB stores and retrieves vector embeddings.
5. Data Processing – Processes educational datasets.
6. Review Storage – SQLite stores educator reviews.

## Application Workflow

Educational Data → Data Processing → Vector Database → RAG Pipeline → AI Insights → Educator Review
