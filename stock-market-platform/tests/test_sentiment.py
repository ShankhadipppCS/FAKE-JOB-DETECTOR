from __future__ import annotations

from app.sentiment.analyzer import SentimentAnalyzer


def test_lexicon_sentiment_detects_positive_text() -> None:
    analyzer = SentimentAnalyzer(use_finbert=False)
    result = analyzer.analyze("NVIDIA shows strong growth and resilient demand")
    assert result.overall_sentiment == "positive"
    assert result.positive_score > result.negative_score

