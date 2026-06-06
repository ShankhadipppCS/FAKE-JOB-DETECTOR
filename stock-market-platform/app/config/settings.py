from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv_env(name: str, default: str) -> list[str]:
    return [item.strip().upper() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AI Stock Market Analytics Platform")
    environment: str = os.getenv("ENVIRONMENT", "local")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://stocks:stocks@postgres:5432/stocks",
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    alpha_vantage_api_key: str | None = os.getenv("ALPHA_VANTAGE_API_KEY")
    polygon_api_key: str | None = os.getenv("POLYGON_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    primary_symbols: list[str] = field(default_factory=lambda: _csv_env("PRIMARY_SYMBOLS", "NVDA,TSLA,AAPL"))
    benchmark_symbol: str = os.getenv("BENCHMARK_SYMBOL", "SPY")
    data_dir: str = os.getenv("DATA_DIR", "data")
    model_registry_dir: str = os.getenv("MODEL_REGISTRY_DIR", "models")
    news_lookback_days: int = int(os.getenv("NEWS_LOOKBACK_DAYS", "14"))


settings = Settings()

