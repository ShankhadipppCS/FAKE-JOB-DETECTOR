from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.config.settings import settings
from app.forecasting.prophet_model import ProphetForecaster
from app.indicators.technical import TechnicalIndicatorEngine
from app.risk.analytics import RiskAnalyticsEngine
from app.sentiment.analyzer import SentimentAnalyzer
from app.services.comparison import ComparativeAnalyticsService
from app.services.data_collection import StockDataCollector
from app.services.insights import AIInsightGenerator
from app.services.news import NewsCollector

st.set_page_config(page_title="AI Stock Market Analytics", page_icon="chart_with_upwards_trend", layout="wide")

collector = StockDataCollector()
indicator_engine = TechnicalIndicatorEngine()
risk_engine = RiskAnalyticsEngine()
sentiment_analyzer = SentimentAnalyzer()
news_collector = NewsCollector()


@st.cache_data(ttl=900)
def load_prices(symbol: str, days: int) -> pd.DataFrame:
    return collector.fetch_historical(symbol, date.today() - timedelta(days=days))


def price_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=frame["date"], open=frame["open"], high=frame["high"], low=frame["low"], close=frame["close"]))
    fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    return fig


st.sidebar.title("AI Market Terminal")
symbols = st.sidebar.multiselect("Symbols", settings.primary_symbols + ["MSFT", "AMZN", "GOOGL", "META"], default=settings.primary_symbols)
symbol = st.sidebar.selectbox("Active stock", symbols or settings.primary_symbols)
days = st.sidebar.slider("History window", 120, 1825, 730)
page = st.sidebar.radio(
    "Workspace",
    [
        "Market Overview",
        "Stock Explorer",
        "Technical Indicators",
        "News & Sentiment",
        "Forecasting",
        "Risk Analysis",
        "Stock Comparison",
        "AI Insights",
    ],
)

st.title("AI Stock Market Analytics & Prediction Platform")

if page == "Market Overview":
    frames = {ticker: load_prices(ticker, days) for ticker in symbols}
    combined = pd.concat(frames.values())
    fig = px.line(combined, x="date", y="close", color="symbol", template="plotly_dark", title="Price Performance")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(combined.groupby("symbol").tail(1), use_container_width=True)

elif page == "Stock Explorer":
    frame = load_prices(symbol, days)
    latest = frame.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"${latest['close']:.2f}")
    c2.metric("Volume", f"{int(latest['volume']):,}")
    c3.metric("High", f"${latest['high']:.2f}")
    c4.metric("Low", f"${latest['low']:.2f}")
    st.plotly_chart(price_chart(frame), use_container_width=True)
    st.download_button("Export CSV", frame.to_csv(index=False), f"{symbol}_prices.csv", "text/csv")

elif page == "Technical Indicators":
    frame = indicator_engine.add_all(load_prices(symbol, days))
    fig = px.line(frame, x="date", y=["close", "sma_20", "sma_50", "ema_12", "ema_26"], template="plotly_dark", title=f"{symbol} Trend Indicators")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.line(frame, x="date", y=["rsi_14", "stoch_k", "stoch_d"], template="plotly_dark", title="Momentum"), use_container_width=True)
    st.dataframe(frame.tail(100), use_container_width=True)

elif page == "News & Sentiment":
    articles = news_collector.fetch_for_symbol(symbol)
    rows = []
    for article in articles:
        sentiment = sentiment_analyzer.analyze(f"{article.headline}. {article.summary}")
        rows.append({**article.model_dump(), **sentiment.model_dump()})
    sentiment_frame = pd.DataFrame(rows)
    st.dataframe(sentiment_frame, use_container_width=True)
    if not sentiment_frame.empty:
        st.plotly_chart(px.bar(sentiment_frame, x="publication_date", y="compound_score", color="overall_sentiment", template="plotly_dark"), use_container_width=True)

elif page == "Forecasting":
    frame = load_prices(symbol, days)
    horizon = st.selectbox("Forecast horizon", [7, 30, 90], index=1)
    forecast = ProphetForecaster().forecast(frame, horizon)
    fig = px.line(frame.tail(250), x="date", y="close", template="plotly_dark", title=f"{symbol} Forecast")
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower", line=dict(dash="dot")))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast, use_container_width=True)

elif page == "Risk Analysis":
    frame = load_prices(symbol, days)
    risk = risk_engine.calculate(frame, load_prices(settings.benchmark_symbol, days))
    st.json(risk.model_dump())
    returns = frame["close"].pct_change().dropna()
    st.plotly_chart(px.histogram(returns, nbins=60, template="plotly_dark", title="Daily Return Distribution"), use_container_width=True)

elif page == "Stock Comparison":
    frames = {ticker: load_prices(ticker, days) for ticker in symbols}
    ranking = ComparativeAnalyticsService().compare(frames)
    st.dataframe(ranking, use_container_width=True)
    st.plotly_chart(px.bar(ranking, x="symbol", y=["total_return", "annual_volatility", "sentiment_score"], barmode="group", template="plotly_dark"), use_container_width=True)

elif page == "AI Insights":
    frame = load_prices(symbol, days)
    risk = risk_engine.calculate(frame)
    article = news_collector.fetch_for_symbol(symbol)[0]
    sentiment = sentiment_analyzer.analyze(article.headline)
    forecast = ProphetForecaster().forecast(frame, 30)
    forecast_return = float(forecast["yhat"].iloc[-1] / frame["close"].iloc[-1] - 1)
    insight = AIInsightGenerator().generate(symbol, risk, sentiment, forecast_return)
    st.markdown(insight)
    st.download_button("Download Report", insight, f"{symbol}_ai_report.txt", "text/plain")
