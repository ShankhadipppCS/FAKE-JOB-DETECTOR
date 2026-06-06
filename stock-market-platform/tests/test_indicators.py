from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.indicators.technical import TechnicalIndicatorEngine


def sample_prices(rows: int = 120) -> pd.DataFrame:
    dates = [date.today() - timedelta(days=rows - idx) for idx in range(rows)]
    close = pd.Series(range(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "symbol": "AAPL",
            "date": dates,
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_add_all_indicators_returns_expected_columns() -> None:
    features = TechnicalIndicatorEngine().add_all(sample_prices())
    expected = {"sma_20", "ema_12", "macd", "rsi_14", "stoch_k", "bb_upper", "atr_14", "obv", "vwap"}
    assert expected.issubset(features.columns)
    assert features["rsi_14"].between(0, 100).all()

