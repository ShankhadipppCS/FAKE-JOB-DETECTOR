from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.models.schemas import NewsArticle
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NewsCollector:
    lookback_days: int = 14

    def fetch_for_symbol(self, symbol: str) -> list[NewsArticle]:
        symbol = symbol.upper()
        articles: list[NewsArticle] = []
        for fetcher in (self._fetch_yahoo_news, self._fallback_news):
            try:
                articles.extend(fetcher(symbol))
            except Exception as exc:
                logger.warning("News fetcher failed for %s: %s", symbol, exc)
        seen: set[str] = set()
        deduped: list[NewsArticle] = []
        for article in articles:
            key = str(article.url)
            if key not in seen:
                deduped.append(article)
                seen.add(key)
        return deduped

    def _fetch_yahoo_news(self, symbol: str) -> list[NewsArticle]:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        rows = getattr(ticker, "news", []) or []
        articles: list[NewsArticle] = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        for row in rows:
            published = datetime.fromtimestamp(row.get("providerPublishTime", 0), tz=timezone.utc)
            if published < cutoff:
                continue
            articles.append(
                NewsArticle(
                    symbol=symbol,
                    headline=row.get("title", ""),
                    summary=row.get("summary", ""),
                    publication_date=published,
                    source=row.get("publisher", "Yahoo Finance"),
                    url=row.get("link", f"https://finance.yahoo.com/quote/{symbol}"),
                )
            )
        return articles

    def _fallback_news(self, symbol: str) -> list[NewsArticle]:
        now = datetime.now(timezone.utc)
        templates = {
            "NVDA": "NVIDIA demand outlook remains resilient as AI infrastructure spending expands.",
            "TSLA": "Tesla investors monitor delivery trends, margins, and autonomous driving milestones.",
            "AAPL": "Apple market narrative focuses on services growth, devices, and AI-enabled products.",
        }
        headline = templates.get(symbol, f"{symbol} market update highlights earnings, sentiment, and risk factors.")
        return [
            NewsArticle(
                symbol=symbol,
                headline=headline,
                summary="Generated fallback article for local demos when live news providers are unavailable.",
                publication_date=now,
                source="Local Demo Feed",
                url=f"https://example.com/news/{symbol.lower()}",
            )
        ]

