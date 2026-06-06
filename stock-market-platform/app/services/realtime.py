from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.services.data_collection import StockDataCollector


class RealtimePriceStreamer:
    def __init__(self) -> None:
        self.collector = StockDataCollector()

    async def stream(self, symbol: str, interval_seconds: float = 5.0) -> AsyncIterator[dict[str, float | str]]:
        while True:
            frame = self.collector.fetch_historical(symbol)
            latest = frame.iloc[-1]
            yield {"symbol": symbol.upper(), "price": float(latest["close"]), "date": str(latest["date"])}
            await asyncio.sleep(interval_seconds)

