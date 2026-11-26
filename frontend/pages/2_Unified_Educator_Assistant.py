
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ==================================================
# 1. PROJECT SETUP
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai.rag_pipeline import ask_course

from backend.review_store import (
    initialize_review_database,
    save_review,
    get_review_history
)


# ==================================================
# 2. FILE PATHS
# ==================================================

ANALYTICS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "learning_difficulty_summary.csv"
)


# ==================================================
# 3. STREAMLIT PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Educator Intelligence Platform",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Unified Educator Intelligence Platform")

st.write(
    "Analyse learning difficulties, inspect supporting "
    "evidence, generate AI recommendations and "
    "review reports before accepting them."
)


# ==================================================
# 4. INITIALIZE SQLITE DATABASE
# ==================================================

try:
    initialize_review_database()

except Exception as error:
    st.error(
        f"Unable to initialize review database: {error}"
    )
    st.stop()


# ==================================================
# 5. LOAD LEARNING ANALYTICS
# ==================================================

@st.cache_data
def load_analytics():

    if not ANALYTICS_PATH.exists():
        raise FileNotFoundError(
            "Learning analytics summary is missing."
        )

    return pd.read_csv(ANALYTICS_PATH)


try:
    analytics = load_analytics()

except Exception as error:
    st.error(str(error))
    st.stop()


# ==================================================
# 6. SELECT EDUCATIONAL TOPIC
# ==================================================

st.sidebar.header("Educator Controls")

topic_ids = (
    analytics["topic_id"]
    .dropna()
    .unique()
    .tolist()
)

if not topic_ids:
    st.warning("No learning topics are available.")
    st.stop()


topic_names = {}

for _, row in analytics.iterrows():

    topic_id = str(row["topic_id"])

    topic_names[topic_id] = str(
        row.get("topic_name", topic_id)
    )


selected_topic = st.sidebar.selectbox(
    "Select learning topic",
    options=topic_ids,
    format_func=lambda topic: (
        f"{topic_names.get(topic, topic)} ({topic})"
    )
)


# ==================================================
# 7. DISPLAY LEARNING ANALYTICS
# ==================================================

st.subheader("1. Learning Performance")

selected_data = analytics[
    analytics["topic_id"] == selected_topic
]

st.dataframe(
    selected_data,
    width="stretch",
    hide_index=True
)


# ==================================================
# 8. EDUCATOR QUESTION
# ==================================================

st.subheader("2. Generate AI Learning Insights")

default_question = (
    f"Why are students struggling with "
    f"{topic_names[selected_topic]}? "
    f"Use quiz analytics, student comments, "
    f"video transcripts and course documents. "
    f"Recommend practical teaching improvements."
)

question = st.text_area(
    "Ask a question",
    value=default_question,
    height=120,
    key=f"educator_question_{selected_topic}"
)

generate_clicked = st.button(
    "Generate Learning Insights",
    type="primary"
)


# ==================================================
# 9. GENERATE AI REPORT
# ==================================================

if generate_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Retrieving evidence and generating report..."
        ):

            try:

                result = ask_course(
                    question=question.strip(),
                    topic_id=selected_topic
                )

                st.session_state[
                    "educator_review_report"
                ] = {
                    "topic_id": selected_topic,
                    "question": result["question"],
                    "answer": result["answer"],
                    "sources": result["sources"]
                }

                st.session_state[
                    "educator_review_saved"
                ] = False

                st.success(
                    "AI report generated. "
                    "Citation IDs were validated."
                )

            except Exception as error:

                st.error(
                    "Unable to generate AI report."
                )

                st.caption(str(error))


# ==================================================
# 10. DISPLAY GENERATED REPORT
# ==================================================

report = st.session_state.get(
    "educator_review_report"
)

report_is_current = (
    report is not None
    and report["topic_id"] == selected_topic
)

if report_is_current:

    st.divider()

    st.subheader("3. AI Learning Insights")

    st.info(
        "This report was generated by AI. "
        "Citation IDs have been checked, but "
        "the accuracy of each claim still "
        "requires educator review."
    )

    st.markdown(
        report["answer"]
    )


    # ==============================================
    # 11. SUPPORTING EVIDENCE
    # ==============================================

    st.subheader("4. Supporting Evidence")

    sources = report["sources"]

    st.caption(
        f"{len(sources)} evidence records retrieved"
    )

    for source in sources:

        reference = source.get(
            "reference", "Unknown"
        )

        modality = source.get(
            "modality", "unknown"
        )

        location = source.get(
            "source_location", "Unknown source"
        )

        heading = (
            f"{reference} | "
            f"{modality.replace('_', ' ').title()} | "
            f"{location}"
        )

        with st.expander(heading):

            st.write(
                source.get("text", "")
            )

            st.caption(
                f"Source ID: {source.get('source_id', '')}"
            )

            start_seconds = source.get(
                "start_seconds"
            )

            if start_seconds is not None:

                minutes = int(start_seconds // 60)
                seconds = int(start_seconds % 60)

                st.info(
                    f"Video timestamp: "
                    f"{minutes:02d}:{seconds:02d}"
                )


    # ==============================================
    # 12. EDUCATOR REVIEW FORM
    # ==============================================

    st.divider()

    st.subheader("5. Educator Review and Approval")

    st.write(
        "Review the AI report and supporting evidence "
        "before saving your decision."
    )

    already_saved = st.session_state.get(
        "educator_review_saved",
        False
    )

    if already_saved:

        st.success(
            "Your decision for this generated report "
            "has already been saved."
        )

    else:

        with st.form(
            "educator_approval_form",
            clear_on_submit=False
        ):

            decision = st.radio(
                "Select your review decision",
                options=[
                    "Approve",
                    "Request changes",
                    "Reject"
                ]
            )

            reviewer_notes = st.text_area(
                "Reviewer comments",
                placeholder=(
                    "Explain your decision. "
                    "Mention any improvements needed."
                ),
                height=120
            )

            submitted = st.form_submit_button(
                "Save Review Decision",
                type="primary"
            )

        if submitted:

            if (
                decision != "Approve"
                and not reviewer_notes.strip()
            ):

                st.warning(
                    "Please explain why the report "
                    "requires changes or was rejected."
                )

            else:

                try:

                    save_review(
                        topic_id=report["topic_id"],
                        question=report["question"],
                        answer=report["answer"],
                        sources=report["sources"],
                        decision=decision,
                        notes=reviewer_notes.strip()
                    )

                    st.session_state[
                        "educator_review_saved"
                    ] = True

                    st.success(
                        "Review decision saved "
                        "successfully to SQLite."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "Unable to save review decision."
                    )

                    st.caption(str(error))


# ==================================================
# 13. REVIEW HISTORY
# ==================================================

st.divider()

st.subheader("6. Educator Review History")

try:

    history = get_review_history()

    if isinstance(history, pd.DataFrame):
        history_df = history.copy()

    elif isinstance(history, list):
        history_df = pd.DataFrame(history)

    elif isinstance(history, dict):
        history_df = pd.DataFrame(
            history.get("reviews", [])
        )

    else:
        history_df = pd.DataFrame()

    if history_df.empty:

        st.info(
            "No educator review decisions "
            "have been saved yet."
        )

    else:

        if "topic_id" in history_df.columns:

            filtered_history = history_df[
                history_df["topic_id"]
                == selected_topic
            ]

        else:
            filtered_history = history_df

        st.metric(
            "Total saved reviews",
            len(history_df)
        )

        st.write(
            "Review decisions for the selected topic:"
        )

        if filtered_history.empty:

            st.info(
                "No reviews found for this topic."
            )

        else:

            st.dataframe(
                filtered_history,
                width="stretch",
                hide_index=True
            )

except Exception as error:

    st.error(
        "Unable to load review history."
    )

    st.caption(str(error))


# ==================================================
# 14. FOOTER
# ==================================================

st.divider()

st.caption(
    "Coursera Multimodal Intelligence Platform | "
    "AI Insights and Human Review"
)
