from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Alert:
    symbol: str
    condition: str
    threshold: float
    channel: str = "email"


class AlertService:
    def evaluate(self, symbol: str, latest_price: float, alerts: list[Alert]) -> list[str]:
        triggered: list[str] = []
        for alert in alerts:
            if alert.symbol.upper() != symbol.upper():
                continue
            if alert.condition == "above" and latest_price > alert.threshold:
                triggered.append(f"{symbol} crossed above {alert.threshold}")
            if alert.condition == "below" and latest_price < alert.threshold:
                triggered.append(f"{symbol} crossed below {alert.threshold}")
        return triggered

    def send_email(self, message: str) -> bool:
        return bool(message)

    def send_telegram(self, message: str) -> bool:
        return bool(message)

