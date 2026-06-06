from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import SentimentResult

POSITIVE_TERMS = {
    "beat",
    "bullish",
    "growth",
    "resilient",
    "strong",
    "upgrade",
    "outperform",
    "profit",
    "demand",
    "expands",
}
NEGATIVE_TERMS = {
    "bearish",
    "cut",
    "decline",
    "downgrade",
    "loss",
    "miss",
    "risk",
    "weak",
    "volatility",
    "uncertainty",
}
STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "for", "as", "on", "in", "with", "is"}


@dataclass
class SentimentAnalyzer:
    use_finbert: bool = True

    def clean_text(self, text: str) -> str:
        text = re.sub(r"https?://\S+", "", text.lower())
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [self._lemmatize(token) for token in text.split() if token not in STOPWORDS]
        return " ".join(tokens)

    def analyze(self, text: str) -> SentimentResult:
        if self.use_finbert:
            result = self._finbert(text)
            if result is not None:
                return result
        return self._vader_or_lexicon(text)

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(text) for text in texts]

    def _lemmatize(self, token: str) -> str:
        for suffix in ("ing", "ed", "s"):
            if len(token) > 4 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    def _vader_or_lexicon(self, text: str) -> SentimentResult:
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer

            scores = SentimentIntensityAnalyzer().polarity_scores(text)
            compound = scores["compound"]
            return self._to_result(scores["pos"], scores["neg"], scores["neu"], compound)
        except Exception:
            clean = self.clean_text(text)
            tokens = set(clean.split())
            positive = len(tokens & POSITIVE_TERMS)
            negative = len(tokens & NEGATIVE_TERMS)
            total = max(positive + negative, 1)
            compound = (positive - negative) / total
            neutral = 1 - min((positive + negative) / max(len(tokens), 1), 1)
            return self._to_result(max(compound, 0), max(-compound, 0), neutral, compound)

    def _finbert(self, text: str) -> SentimentResult | None:
        try:
            from transformers import pipeline

            classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            raw = classifier(text[:512])[0]
            label = raw["label"].lower()
            score = float(raw["score"])
            positive = score if "positive" in label else 0.0
            negative = score if "negative" in label else 0.0
            neutral = score if "neutral" in label else 1.0 - max(positive, negative)
            compound = positive - negative
            return self._to_result(positive, negative, neutral, compound)
        except Exception:
            return None

    @staticmethod
    def _to_result(positive: float, negative: float, neutral: float, compound: float) -> SentimentResult:
        total = max(positive + negative + neutral, 1e-9)
        positive, negative, neutral = positive / total, negative / total, neutral / total
        label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
        return SentimentResult(
            positive_score=round(float(positive), 4),
            negative_score=round(float(negative), 4),
            neutral_score=round(float(neutral), 4),
            compound_score=round(float(compound), 4),
            overall_sentiment=label,
        )

