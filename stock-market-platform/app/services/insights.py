from __future__ import annotations

from app.models.schemas import RiskMetrics, SentimentResult


class AIInsightGenerator:
    def generate(
        self,
        symbol: str,
        risk: RiskMetrics,
        sentiment: SentimentResult | None = None,
        forecast_return: float | None = None,
    ) -> str:
        tone = sentiment.overall_sentiment if sentiment else "neutral"
        direction = "upside potential" if forecast_return and forecast_return > 0 else "limited near-term upside"
        risk_phrase = risk.risk_category.lower()
        if risk.sharpe_ratio > 1:
            quality = "attractive risk-adjusted returns"
        elif risk.maximum_drawdown < -0.35:
            quality = "meaningful drawdown pressure"
        else:
            quality = "mixed risk-adjusted signals"
        return (
            f"{symbol.upper()} currently shows {tone} news sentiment, {risk_phrase}, and {quality}. "
            f"The forecast profile indicates {direction}; position sizing should account for volatility, "
            "beta exposure, and recent trend confirmation."
        )

    def comparative_report(self, summaries: list[str]) -> str:
        return "\n\n".join(summaries)

