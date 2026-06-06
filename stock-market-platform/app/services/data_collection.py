from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.config.settings import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StockDataCollector:
    data_dir: Path = Path(settings.data_dir)

    def fetch_historical(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        source: str = "yahoo",
    ) -> pd.DataFrame:
        start = start or date.today() - timedelta(days=365 * 3)
        end = end or date.today()
        symbol = symbol.upper()
        try:
            if source == "yahoo":
                return self._fetch_yahoo(symbol, start, end)
        except Exception as exc:
            logger.warning("Market data fetch failed for %s; using synthetic fallback: %s", symbol, exc)
        return self._synthetic_prices(symbol, start, end)

    def incremental_update(self, symbol: str, existing: pd.DataFrame | None = None) -> pd.DataFrame:
        if existing is None or existing.empty:
            return self.fetch_historical(symbol)
        last_date = pd.to_datetime(existing["date"]).max().date()
        fresh = self.fetch_historical(symbol, start=last_date + timedelta(days=1))
        return self.validate_prices(pd.concat([existing, fresh], ignore_index=True))

    def validate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["symbol", "date", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing price columns: {missing}")
        clean = df.copy()
        clean["date"] = pd.to_datetime(clean["date"]).dt.date
        clean = clean.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
        clean[["open", "high", "low", "close"]] = clean[["open", "high", "low", "close"]].ffill().bfill()
        clean["volume"] = clean["volume"].fillna(0).clip(lower=0).astype(int)
        return clean[required].reset_index(drop=True)

    def save_local(self, df: pd.DataFrame, layer: str = "raw") -> Path:
        if df.empty:
            raise ValueError("Cannot save empty market data")
        symbol = str(df["symbol"].iloc[0]).upper()
        path = self.data_dir / layer / f"{symbol}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def _fetch_yahoo(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        data = yf.download(symbol, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), progress=False)
        if data.empty:
            raise ValueError(f"No Yahoo Finance rows returned for {symbol}")
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [column[0].lower() for column in data.columns]
        else:
            data.columns = [str(column).lower() for column in data.columns]
        data["symbol"] = symbol
        data = data.rename(columns={"adj close": "adj_close"})
        return self.validate_prices(data[["symbol", "date", "open", "high", "low", "close", "volume"]])

    def _synthetic_prices(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        dates = pd.bdate_range(start, end)
        base = {"NVDA": 120.0, "TSLA": 180.0, "AAPL": 210.0}.get(symbol, 100.0)
        drift = {"NVDA": 0.0012, "TSLA": 0.0007, "AAPL": 0.0006}.get(symbol, 0.0005)
        shocks = rng.normal(drift, 0.025, len(dates))
        close = base * np.exp(np.cumsum(shocks))
        open_ = close * (1 + rng.normal(0, 0.006, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, len(dates)))
        volume = rng.integers(20_000_000, 180_000_000, len(dates))
        return self.validate_prices(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates.date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )

