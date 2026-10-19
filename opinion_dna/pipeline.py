
from opinion_dna.sentiment import analyze_sentiment
from opinion_dna.aspects import analyze_aspects
from opinion_dna.emotions import analyze_emotion


def analyze_review(review):
    """Run all Opinion DNA analyses on one customer review."""

    if not isinstance(review, str) or not review.strip():
        raise ValueError("Review text cannot be empty.")

    review = review.strip()

    # Run the three existing analysis functions.
    overall = analyze_sentiment(review)
    aspects = analyze_aspects(review)
    emotion = analyze_emotion(review)

    # Return all results in one structured dictionary.
    return {
        "review": review,
        "overall": overall,
        "aspects": aspects,
        "emotion": emotion,
    }
