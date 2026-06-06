from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.models.schemas import RiskMetrics


@dataclass
class RiskAnalyticsEngine:
    risk_free_rate: float = 0.04
    trading_days: int = 252

    def calculate(self, prices: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> RiskMetrics:
        symbol = str(prices["symbol"].iloc[0]).upper()
        returns = prices["close"].pct_change().dropna()
        benchmark_returns = benchmark["close"].pct_change().dropna() if benchmark is not None and not benchmark.empty else returns
        volatility = float(returns.std() * np.sqrt(self.trading_days))
        expected_return = float(returns.mean() * self.trading_days)
        downside = returns[returns < 0]
        sharpe = self._safe_ratio(expected_return - self.risk_free_rate, volatility)
        sortino = self._safe_ratio(expected_return - self.risk_free_rate, float(downside.std() * np.sqrt(self.trading_days)))
        beta = self._beta(returns, benchmark_returns)
        drawdown = self.maximum_drawdown(prices["close"])
        var = float(np.percentile(returns, 5))
        score = self.risk_score(volatility, abs(beta), abs(drawdown), abs(var))
        return RiskMetrics(
            symbol=symbol,
            volatility=volatility,
            beta=beta,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            maximum_drawdown=drawdown,
            value_at_risk=var,
            expected_return=expected_return,
            risk_score=score,
            risk_category=self.risk_category(score),
        )

    def optimize_portfolio(self, price_frames: dict[str, pd.DataFrame]) -> dict[str, float | dict[str, float]]:
        closes = pd.DataFrame({symbol: frame.set_index("date")["close"] for symbol, frame in price_frames.items()}).dropna()
        returns = closes.pct_change().dropna()
        if returns.empty:
            raise ValueError("Not enough overlapping data for portfolio optimization")
        mean_returns = returns.mean() * self.trading_days
        cov = returns.cov() * self.trading_days
        inv_cov = np.linalg.pinv(cov.to_numpy())
        raw = inv_cov @ mean_returns.to_numpy()
        weights = raw / raw.sum() if raw.sum() else np.repeat(1 / len(mean_returns), len(mean_returns))
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()
        expected = float(np.dot(weights, mean_returns))
        variance = float(weights.T @ cov.to_numpy() @ weights)
        return {
            "expected_return": expected,
            "portfolio_variance": variance,
            "weights": {symbol: float(weight) for symbol, weight in zip(mean_returns.index, weights, strict=False)},
        }

    @staticmethod
    def maximum_drawdown(close: pd.Series) -> float:
        cumulative_max = close.cummax()
        drawdown = close / cumulative_max - 1
        return float(drawdown.min())

    @staticmethod
    def _beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2 or aligned.iloc[:, 1].var() == 0:
            return 1.0
        return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator and not np.isnan(denominator) else 0.0

    @staticmethod
    def risk_score(volatility: float, beta: float, drawdown: float, var: float) -> float:
        score = 100 * (0.4 * min(volatility / 0.8, 1) + 0.2 * min(beta / 2, 1) + 0.25 * min(drawdown / 0.7, 1) + 0.15 * min(var / 0.08, 1))
        return round(float(score), 2)

    @staticmethod
    def risk_category(score: float) -> str:
        if score < 35:
            return "Low Risk"
        if score < 70:
            return "Medium Risk"
        return "High Risk"

