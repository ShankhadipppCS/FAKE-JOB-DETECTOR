from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, Query, WebSocket

from app.config.settings import settings
from app.forecasting.deep_learning import LSTMForecaster
from app.forecasting.prophet_model import ProphetForecaster
from app.indicators.technical import TechnicalIndicatorEngine
from app.models.schemas import ApiResponse, ForecastRequest
from app.risk.analytics import RiskAnalyticsEngine
from app.sentiment.analyzer import SentimentAnalyzer
from app.services.comparison import ComparativeAnalyticsService
from app.services.data_collection import StockDataCollector
from app.services.insights import AIInsightGenerator
from app.services.news import NewsCollector
from app.services.realtime import RealtimePriceStreamer
from app.services.screener import AIStockScreener

router = APIRouter()
collector = StockDataCollector()
indicator_engine = TechnicalIndicatorEngine()
risk_engine = RiskAnalyticsEngine()
news_collector = NewsCollector(settings.news_lookback_days)
sentiment_analyzer = SentimentAnalyzer()


@router.get("/health", response_model=ApiResponse)
def health() -> ApiResponse:
    return ApiResponse(data={"status": "healthy", "environment": settings.environment})


@router.get("/stocks/{symbol}/prices", response_model=ApiResponse)
def prices(symbol: str, days: int = Query(365, ge=30, le=3650)) -> ApiResponse:
    frame = collector.fetch_historical(symbol, date.today() - timedelta(days=days))
    return ApiResponse(data=frame.to_dict(orient="records"))


@router.get("/stocks/{symbol}/indicators", response_model=ApiResponse)
def indicators(symbol: str, days: int = Query(365, ge=90, le=3650)) -> ApiResponse:
    frame = collector.fetch_historical(symbol, date.today() - timedelta(days=days))
    features = indicator_engine.add_all(frame)
    collector.save_local(features, "features")
    return ApiResponse(data=features.tail(200).to_dict(orient="records"))


@router.get("/stocks/{symbol}/news", response_model=ApiResponse)
def news(symbol: str) -> ApiResponse:
    articles = news_collector.fetch_for_symbol(symbol)
    enriched = []
    for article in articles:
        sentiment = sentiment_analyzer.analyze(f"{article.headline}. {article.summary}")
        enriched.append({**article.model_dump(), "sentiment": sentiment.model_dump()})
    return ApiResponse(data=enriched)


@router.post("/forecast", response_model=ApiResponse)
def forecast(request: ForecastRequest) -> ApiResponse:
    frame = collector.fetch_historical(request.symbol)
    if request.model == "lstm":
        result = LSTMForecaster().train_predict(frame)
        return ApiResponse(data=result)
    forecast_frame = ProphetForecaster().forecast(frame, request.horizon_days)
    return ApiResponse(data=forecast_frame.to_dict(orient="records"))


@router.get("/stocks/{symbol}/risk", response_model=ApiResponse)
def risk(symbol: str) -> ApiResponse:
    frame = collector.fetch_historical(symbol)
    benchmark = collector.fetch_historical(settings.benchmark_symbol)
    return ApiResponse(data=risk_engine.calculate(frame, benchmark).model_dump())


@router.get("/comparison", response_model=ApiResponse)
def comparison(symbols: str = Query(default=",".join(settings.primary_symbols))) -> ApiResponse:
    selected = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
    frames = {symbol: collector.fetch_historical(symbol) for symbol in selected}
    sentiment_scores: dict[str, float] = {}
    for symbol in selected:
        articles = news_collector.fetch_for_symbol(symbol)
        sentiments = [sentiment_analyzer.analyze(article.headline).compound_score for article in articles]
        sentiment_scores[symbol] = float(pd.Series(sentiments).mean()) if sentiments else 0.0
    service = ComparativeAnalyticsService()
    ranking = service.compare(frames, sentiment_scores)
    return ApiResponse(data={"ranking": ranking.to_dict(orient="records"), "summary": service.executive_summary(ranking)})


@router.get("/insights/{symbol}", response_model=ApiResponse)
def insights(symbol: str) -> ApiResponse:
    frame = collector.fetch_historical(symbol)
    risk = risk_engine.calculate(frame)
    article = news_collector.fetch_for_symbol(symbol)[0]
    sentiment = sentiment_analyzer.analyze(f"{article.headline}. {article.summary}")
    forecast_frame = ProphetForecaster().forecast(frame, 30)
    forecast_return = float(forecast_frame["yhat"].iloc[-1] / frame["close"].iloc[-1] - 1)
    insight = AIInsightGenerator().generate(symbol, risk, sentiment, forecast_return)
    return ApiResponse(data={"insight": insight})


@router.get("/screener", response_model=ApiResponse)
def screener(symbols: str = Query(default=",".join(settings.primary_symbols))) -> ApiResponse:
    selected = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
    frames = {symbol: collector.fetch_historical(symbol) for symbol in selected}
    result = AIStockScreener().screen(frames)
    return ApiResponse(data=result.to_dict(orient="records"))


@router.websocket("/stream/{symbol}")
async def price_stream(websocket: WebSocket, symbol: str) -> None:
    await websocket.accept()
    streamer = RealtimePriceStreamer()
    async for tick in streamer.stream(symbol, interval_seconds=5):
        await websocket.send_json(tick)

