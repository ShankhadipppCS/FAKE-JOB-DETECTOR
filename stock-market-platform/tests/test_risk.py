from __future__ import annotations

from tests.test_indicators import sample_prices
from app.risk.analytics import RiskAnalyticsEngine


def test_risk_metrics_are_bounded() -> None:
    risk = RiskAnalyticsEngine().calculate(sample_prices())
    assert 0 <= risk.risk_score <= 100
    assert risk.risk_category in {"Low Risk", "Medium Risk", "High Risk"}
    assert risk.symbol == "AAPL"


def test_portfolio_optimization_weights_sum_to_one() -> None:
    frame = sample_prices()
    result = RiskAnalyticsEngine().optimize_portfolio({"AAPL": frame, "NVDA": frame.assign(symbol="NVDA", close=frame["close"] * 1.1)})
    assert abs(sum(result["weights"].values()) - 1) < 1e-6

