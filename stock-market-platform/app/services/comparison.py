from __future__ import annotations

import pandas as pd

from app.risk.analytics import RiskAnalyticsEngine


class ComparativeAnalyticsService:
    def __init__(self) -> None:
        self.risk_engine = RiskAnalyticsEngine()

    def compare(self, frames: dict[str, pd.DataFrame], sentiment: dict[str, float] | None = None) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        sentiment = sentiment or {}
        for symbol, frame in frames.items():
            returns = frame["close"].pct_change().dropna()
            risk = self.risk_engine.calculate(frame)
            total_return = float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1)
            rows.append(
                {
                    "symbol": symbol,
                    "total_return": total_return,
                    "annual_volatility": risk.volatility,
                    "sharpe_ratio": risk.sharpe_ratio,
                    "max_drawdown": risk.maximum_drawdown,
                    "risk_score": risk.risk_score,
                    "sentiment_score": sentiment.get(symbol, 0.0),
                }
            )
        table = pd.DataFrame(rows)
        table["composite_score"] = (
            table["total_return"].rank(pct=True) * 0.35
            + table["sharpe_ratio"].rank(pct=True) * 0.25
            + (1 - table["risk_score"].rank(pct=True)) * 0.25
            + table["sentiment_score"].rank(pct=True) * 0.15
        )
        return table.sort_values("composite_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def executive_summary(ranking: pd.DataFrame) -> str:
        leader = ranking.iloc[0]
        laggard = ranking.iloc[-1]
        return (
            f"{leader['symbol']} ranks highest on the composite score, supported by relative performance, "
            f"risk-adjusted returns, and sentiment. {laggard['symbol']} ranks lowest and should be reviewed "
            "for volatility, drawdown, and forecast uncertainty before portfolio allocation."
        )

