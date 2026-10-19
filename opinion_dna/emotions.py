
from transformers import pipeline

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

_emotion_classifier = None


def load_emotion_model():
    """Load the emotion model once and reuse it."""
    global _emotion_classifier

    if _emotion_classifier is None:
        _emotion_classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None,
            device=-1,
        )

    return _emotion_classifier


def analyze_emotion(text):
    """Predict emotions expressed in a review."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review text cannot be empty.")

    classifier = load_emotion_model()
    predictions = classifier(text.strip(), truncation=True)[0]

    best = max(predictions, key=lambda item: item["score"])

    return {
        "emotion": best["label"].lower(),
        "score": float(best["score"]),
        "all_emotions": [
            {
                "emotion": item["label"].lower(),
                "score": float(item["score"]),
            }
            for item in sorted(
                predictions,
                key=lambda item: item["score"],
                reverse=True,
            )
        ],
    }
