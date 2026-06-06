from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd
from sqlalchemy import Column, Date, DateTime, Float, Integer, MetaData, String, Table, UniqueConstraint, create_engine
from sqlalchemy.engine import Engine

from app.config.settings import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
metadata = MetaData()

stock_prices = Table(
    "stock_prices",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String(16), nullable=False, index=True),
    Column("date", Date, nullable=False, index=True),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", Integer, nullable=False),
    UniqueConstraint("symbol", "date", name="uq_stock_prices_symbol_date"),
)

news_articles = Table(
    "news_articles",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String(16), nullable=False, index=True),
    Column("headline", String(512), nullable=False),
    Column("summary", String(4000), nullable=False, default=""),
    Column("publication_date", DateTime, nullable=False, index=True),
    Column("source", String(128), nullable=False),
    Column("url", String(1024), nullable=False),
    UniqueConstraint("symbol", "url", name="uq_news_symbol_url"),
)


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or settings.database_url, pool_pre_ping=True)


def create_schema(engine: Engine | None = None) -> None:
    metadata.create_all(engine or get_engine())


@contextmanager
def db_engine(url: str | None = None) -> Iterator[Engine]:
    engine = get_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()


def upsert_dataframe(df: pd.DataFrame, table_name: str, engine: Engine) -> int:
    if df.empty:
        return 0
    df.to_sql(table_name, engine, if_exists="append", index=False, method="multi")
    return len(df)
