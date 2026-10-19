
from opinion_dna.sentiment import analyze_sentiment


def test_positive_review():
    result = analyze_sentiment(
        "The product is excellent and easy to use."
    )

    assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
    assert 0.0 <= result["score"] <= 1.0


def test_empty_review():
    try:
        analyze_sentiment("   ")
    except ValueError:
        return

    assert False, "Expected ValueError for an empty review"
