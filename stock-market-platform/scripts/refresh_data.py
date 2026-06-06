from __future__ import annotations

from app.config.settings import settings
from app.indicators.technical import TechnicalIndicatorEngine
from app.services.data_collection import StockDataCollector


def main() -> None:
    collector = StockDataCollector()
    indicators = TechnicalIndicatorEngine()
    for symbol in settings.primary_symbols:
        prices = collector.fetch_historical(symbol)
        collector.save_local(prices, "raw")
        collector.save_local(indicators.add_all(prices), "features")


if __name__ == "__main__":
    main()

