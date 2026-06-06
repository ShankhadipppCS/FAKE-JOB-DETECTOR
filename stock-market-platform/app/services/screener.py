from __future__ import annotations

import pandas as pd

from app.indicators.technical import TechnicalIndicatorEngine
from app.risk.analytics import RiskAnalyticsEngine


class AIStockScreener:
    def __init__(self) -> None:
        self.indicators = TechnicalIndicatorEngine()
        self.risk = RiskAnalyticsEngine()

    def screen(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for symbol, frame in frames.items():
            features = self.indicators.add_all(frame)
            latest = features.iloc[-1]
            risk = self.risk.calculate(frame)
            bullish = latest["close"] > latest["sma_50"] and latest["macd"] > latest["macd_signal"] and latest["rsi_14"] < 75
            rows.append(
                {
                    "symbol": symbol,
                    "close": latest["close"],
                    "bullish_signal": bool(bullish),
                    "risk_score": risk.risk_score,
                    "score": (1 if bullish else 0) * 60 + max(0, 40 - risk.risk_score * 0.4),
                }
            )
        return pd.DataFrame(rows).sort_values("score", ascending=False)

