from transformers import pipeline
import spacy

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"

# Load spaCy once for extracting candidate aspect phrases.
_nlp = spacy.load("en_core_web_sm")

# Load the sentiment classifier only when needed.
_aspect_classifier = None


def load_aspect_classifier():
    """Load the ABSA classifier once and reuse it."""
    global _aspect_classifier

    if _aspect_classifier is None:
        _aspect_classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            device=-1,
        )

    return _aspect_classifier


def extract_aspects(review):
    """Extract candidate aspect phrases from a review."""
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Review text cannot be empty.")

    doc = _nlp(review.strip())
    aspects = []

    for chunk in doc.noun_chunks:
        aspect_doc = _nlp(chunk.text.strip())

        # FIX: Properly indented inside the for loop
        while len(aspect_doc) > 0 and aspect_doc[0].lower_ in {"the", "a", "an"}:
            aspect_doc = aspect_doc[1:]

        aspect = aspect_doc.text.strip().lower()

        if aspect and aspect not in aspects:
            aspects.append(aspect)

    return aspects


def analyze_aspects(review):
    """Extract aspects and predict sentiment for each aspect."""
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Review text cannot be empty.")

    classifier = load_aspect_classifier()
    aspects = extract_aspects(review)

    results = []

    for aspect in aspects:
        predictions = classifier(
            review.strip(),
            text_pair=aspect,
            top_k=None,
        )

        best = max(predictions, key=lambda item: item["score"])

        results.append({
            "aspect": aspect,
            "sentiment": best["label"].upper(),
            "score": float(best["score"]),
        })

    return results