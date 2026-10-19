import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from opinion_dna.pipeline import analyze_review

st.set_page_config(
    page_title="Opinion DNA",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Opinion DNA")
st.subheader("Customer Opinion Intelligence")

st.write(
    "Analyze customer reviews using sentiment analysis, "
    "aspect-based sentiment, and emotion detection."
)

st.divider()

st.header("Review Analysis")

review = st.text_area(
    "Enter a customer review",
    placeholder=(
        "Example: The camera quality is excellent, "
        "but the battery drains quickly."
    ),
    height=150,
)

if st.button("Analyze Opinion", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            with st.spinner("Analyzing your review..."):
                result = analyze_review(review)

            st.session_state["analysis"] = result

        except Exception as error:
            st.error(f"Analysis failed: {error}")

# Display the latest analysis.
if "analysis" in st.session_state:
    data = st.session_state["analysis"]
    overall = data["overall"]
    aspects = data["aspects"]
    emotion = data["emotion"]

    st.divider()
    st.header("Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall Sentiment",
        overall["sentiment"].title(),
    )

    col2.metric(
        "Detected Emotion",
        emotion["emotion"].title(),
    )

    col3.metric(
        "Aspects Detected",
        len(aspects),
    )

    st.subheader("Aspect-Based Sentiment")

    if aspects:
        aspect_df = pd.DataFrame([
            {
                "Aspect": item["aspect"],
                "Sentiment": item["sentiment"].title(),
                "Confidence": round(item["score"], 4),
            }
            for item in aspects
        ])

        st.dataframe(
            aspect_df,
            use_container_width=True,
            hide_index=True,
        )

        sentiment_counts = (
            aspect_df["Sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Sentiment by Number of Detected Aspects",
            text="Count",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No aspects were detected in this review.")

    st.subheader("Emotion Predictions")

    emotion_df = pd.DataFrame(emotion["all_emotions"])
    emotion_df["emotion"] = emotion_df["emotion"].str.title()
    emotion_df["score"] = emotion_df["score"].round(4)

    st.dataframe(
        emotion_df.rename(columns={
            "emotion": "Emotion",
            "score": "Confidence",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Create one CSV row per detected aspect.
    export_rows = []

    if aspects:
        for item in aspects:
            export_rows.append({
                "Review": data["review"],
                "Overall Sentiment": overall["sentiment"],
                "Overall Confidence": overall["score"],
                "Emotion": emotion["emotion"],
                "Emotion Confidence": emotion["score"],
                "Aspect": item["aspect"],
                "Aspect Sentiment": item["sentiment"],
                "Aspect Confidence": item["score"],
            })
    else:
        export_rows.append({
            "Review": data["review"],
            "Overall Sentiment": overall["sentiment"],
            "Overall Confidence": overall["score"],
            "Emotion": emotion["emotion"],
            "Emotion Confidence": emotion["score"],
            "Aspect": "",
            "Aspect Sentiment": "",
            "Aspect Confidence": "",
        })

    export_df = pd.DataFrame(export_rows)

    st.download_button(
        label="Download Analysis as CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="opinion_dna_analysis.csv",
        mime="text/csv",
    )

    st.caption(
        "Predictions are model estimates and may be incorrect. "
        "Aspect extraction may miss or misidentify product features."
    )
