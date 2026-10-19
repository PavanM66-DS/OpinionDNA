
from transformers import pipeline

MODEL_NAME = (
    "distilbert/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

_sentiment_pipeline = None


def load_sentiment_model():
    """Load the model once and reuse it."""
    global _sentiment_pipeline

    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            device=-1,
        )

    return _sentiment_pipeline


def analyze_sentiment(text):
    """Predict the overall sentiment of one text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review text cannot be empty.")

    model = load_sentiment_model()
    result = model(text.strip())[0]

    return {
        "sentiment": result["label"],
        "score": float(result["score"]),
    }
