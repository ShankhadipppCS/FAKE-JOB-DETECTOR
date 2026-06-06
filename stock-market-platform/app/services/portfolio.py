from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Position:
    symbol: str
    shares: float
    average_cost: float


class PortfolioTracker:
    def value(self, positions: list[Position], latest_prices: dict[str, float]) -> dict[str, float]:
        market_value = 0.0
        cost_basis = 0.0
        for position in positions:
            price = latest_prices.get(position.symbol.upper(), position.average_cost)
            market_value += position.shares * price
            cost_basis += position.shares * position.average_cost
        return {
            "market_value": market_value,
            "cost_basis": cost_basis,
            "unrealized_pnl": market_value - cost_basis,
            "unrealized_pnl_pct": (market_value / cost_basis - 1) if cost_basis else 0.0,
        }

    def allocation(self, positions: list[Position], latest_prices: dict[str, float]) -> pd.DataFrame:
        rows = []
        total = self.value(positions, latest_prices)["market_value"]
        for position in positions:
            price = latest_prices.get(position.symbol.upper(), position.average_cost)
            market_value = position.shares * price
            rows.append({"symbol": position.symbol.upper(), "market_value": market_value, "weight": market_value / total if total else 0.0})
        return pd.DataFrame(rows)

