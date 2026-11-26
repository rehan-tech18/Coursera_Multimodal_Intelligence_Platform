
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai.rag_pipeline import ask_course
from backend.review_store import (
    save_review,
    get_review_history
)

# 1. Configure the application
st.set_page_config(
    page_title="Coursera Intelligence Platform",
    page_icon="🎓",
    layout="wide"
)

# 2. Locate the project folders
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

# 3. Load our processed analytics
learning_data = pd.read_csv(
    PROCESSED_DIR / "learning_difficulty_summary.csv"
)

comments = pd.read_csv(
    RAW_DIR / "student_comments.csv"
)

# 4. Application heading
st.title("🎓 Coursera Multimodal Intelligence Platform")
st.subheader("Learning Analytics Dashboard")

st.caption("Demonstration using synthetic learner data")

# 5. Display summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Quiz Attempts",
        int(learning_data["total_attempts"].sum())
    )

with col2:
    st.metric(
        "Student Comments",
        len(comments)
    )

with col3:
    st.metric(
        "Topics Analysed",
        len(learning_data)
    )

st.divider()

# 6. Display learning difficulty chart
st.subheader("Topics Students Struggle With Most")

chart_data = learning_data.set_index(
    "topic_name"
)[["incorrect_rate_pct"]]

st.bar_chart(chart_data, horizontal=True)

# 7. Display complete topic-level data
st.subheader("Detailed Learning Analytics")

st.dataframe(
    learning_data[
        [
            "topic_name",
            "total_attempts",
            "incorrect_answers",
            "incorrect_rate_pct",
            "comment_count"
        ]
    ],
    hide_index=True,
    width="stretch"
)

# 8. Show the highest error-rate topic
most_difficult = learning_data.loc[
    learning_data["incorrect_rate_pct"].idxmax()
]

st.warning(
    f"Highest quiz error rate: "
    f"{most_difficult['topic_name']} "
    f"({most_difficult['incorrect_rate_pct']:.2f}%)"
)

# 9. Interactive topic explorer
st.divider()
st.subheader("Explore Student Difficulties")

# Create a mapping between topic IDs and readable names
topic_options = dict(
    zip(
        learning_data["topic_id"],
        learning_data["topic_name"]
    )
)

# Allow the educator to select a topic
selected_topic = st.selectbox(
    "Select a Python topic",
    options=learning_data["topic_id"].tolist(),
    format_func=lambda topic_id: topic_options[topic_id]
)

# Find the selected topic's analytics
selected_data = learning_data[
    learning_data["topic_id"] == selected_topic
].iloc[0]

# Find the student comments for that topic
selected_comments = comments[
    comments["topic_id"] == selected_topic
]

# Display topic metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Incorrect Answer Rate",
        f"{selected_data['incorrect_rate_pct']:.2f}%"
    )

with col2:
    st.metric(
        "Quiz Attempts",
        int(selected_data["total_attempts"])
    )

with col3:
    st.metric(
        "Student Comments",
        len(selected_comments)
    )

# Display student comments
st.subheader("Student Feedback")

if selected_comments.empty:
    st.info("No student comments found for this topic.")

else:
    for _, comment in selected_comments.iterrows():

        with st.expander(
            f"Comment {comment['comment_id']}"
        ):
            st.write(comment["comment_text"])

            st.caption(
                f"Type: {comment['comment_type']}"
            )


# 10. Connect learning difficulties to video lessons
st.divider()
st.subheader("Related Video Evidence")

# Load video topic information
video_topics = pd.read_csv(
    RAW_DIR / "video_topics.csv"
)

# Find selected topic's video metadata
matching_video = video_topics[
    (video_topics["topic_id"] == selected_topic)
    & (video_topics["course_id"] == selected_data["course_id"])
    & (video_topics["video_id"] == selected_data["video_id"])
]

if matching_video.empty:
    st.info("No video segment found for this topic.")

else:
    video_info = matching_video.iloc[0]

    start_time = video_info["approx_start_time"]
    end_time = video_info["approx_end_time"]

    st.write("**Topic:**", video_info["topic_name"])
    st.write("**Approximate start:**", start_time)
    st.write("**Approximate end:**", end_time)

    # Convert HH:MM:SS into seconds
    def time_to_seconds(time_text):
        hours, minutes, seconds = map(
            int,
            str(time_text).split(":")
        )

        return hours * 3600 + minutes * 60 + seconds

    start_seconds = time_to_seconds(start_time)

    # Find the MP4 video
    video_files = list(RAW_DIR.glob("*.mp4"))

    if video_files:
        video_path = video_files[0]

        st.video(
            str(video_path),
            start_time=start_seconds
        )

    else:
        st.warning("No MP4 video found in data/raw.")

    st.caption(
        "Video timestamps are approximate and aligned "
        "using the generated lesson transcript."
    )

# 11. AI EDUCATOR ASSISTANT
st.divider()

st.header("🤖 AI Educator Assistant")

st.write(
    "Ask questions about learner difficulties, "
    "quiz performance and course evidence."
)

question = st.text_area(
    "Your question",
    value=(
        f"Why are students struggling with "
        f"{topic_options[selected_topic]}? "
        f"Use quiz results, student feedback and "
        f"video transcript evidence."
    ),
    height=120,
    key=f"educator_question_{selected_topic}"
)

# Generate a new report
if st.button("Generate Learning Insights"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Retrieving and analyzing evidence..."):
            try:
                result = ask_course(
                    question,
                    topic_id=selected_topic
                )

                # Preserve the report across Streamlit reruns
                st.session_state["rag_report"] = {
                    "topic_id": selected_topic,
                    "result": result
                }

            except Exception as error:
                st.error(f"Unable to generate insights: {error}")

# Retrieve the last generated report
saved_report = st.session_state.get("rag_report")

# Only show a report for the currently selected topic
if (
    saved_report is not None
    and saved_report["topic_id"] == selected_topic
):
    result = saved_report["result"]

    st.subheader("Learning Insights")
    st.markdown(result["answer"])

    st.divider()
    st.subheader("Supporting Evidence")

    sources = result.get("sources", [])

    if not sources:
        st.info("No supporting evidence was found.")

    else:
        for source in sources:

            with st.expander(
                f"{source['reference']} — "
                f"{source['modality']} — "
                f"{source['source_id']}"
            ):
                st.write("**Topic:**", source["topic_id"])

                st.write(
                    "**Source location:**",
                    source["source_location"]
                )

                start = source.get("start_seconds")
                end = source.get("end_seconds")

                if start is not None and end is not None:
                    start = int(start)
                    end = int(end)

                    st.write(
                        f"**Video timestamp:** "
                        f"{start // 60:02d}:{start % 60:02d} "
                        f"– {end // 60:02d}:{end % 60:02d}"
                    )

                    if source["modality"] == "video_transcript":
                        video_files = list(RAW_DIR.glob("*.mp4"))

                        if video_files:
                            st.video(
                                str(video_files[0]),
                                start_time=start
                            )

                st.write("**Retrieved evidence:**")
                st.write(source["text"])

    # 12. EDUCATOR REVIEW
    st.divider()
    st.header("Educator Review")

    st.write(
        "Review the AI-generated insights and "
        "record your decision."
    )

    with st.form("educator_review_form"):

        decision = st.selectbox(
            "Review decision",
            [
                "Approve",
                "Request changes",
                "Reject"
            ]
        )

        notes = st.text_area(
            "Reviewer comments",
            placeholder="Explain your decision..."
        )

        submitted = st.form_submit_button(
            "Save Review Decision"
        )

        if submitted:

            if decision != "Approve" and not notes.strip():
                st.warning(
                    "Please explain why changes are needed "
                    "or why the report was rejected."
                )

            elif not sources:
                st.warning(
                    "Cannot review a report without evidence."
                )

            else:
                review_id = save_review(
                    topic_id=selected_topic,
                    question=result["question"],
                    answer=result["answer"],
                    sources=sources,
                    decision=decision,
                    notes=notes
                )

                st.success(
                    f"Review #{review_id} saved successfully."
                )

# 13. REVIEW HISTORY
st.divider()
st.header("Educator Review History")

history = get_review_history()

if not history:
    st.info("No educator reviews have been saved yet.")

else:
    for review in history:

        with st.expander(
            f"Review #{review['review_id']} — "
            f"{review['topic_id']} — "
            f"{review['decision']}"
        ):
            st.write("**Decision:**", review["decision"])
            st.write("**Topic:**", review["topic_id"])

            st.write(
                "**Reviewed at (UTC):**",
                review["reviewed_at"]
            )

            st.write(
                "**Reviewer notes:**",
                review["reviewer_notes"] or "None"
            )

            st.write(
                "**Evidence IDs:**",
                review["evidence_ids"]
            )

            st.write("**Original question:**")
            st.write(review["question"])

            st.write("**Saved AI answer:**")
            st.markdown(review["ai_answer"])

st.caption(
    "Prototype using synthetic learner data. "
    "Review decisions are stored locally; "
    "authentication and authorization are not yet enabled."
)
