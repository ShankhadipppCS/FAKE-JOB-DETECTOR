from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProphetForecaster:
    interval_width: float = 0.8

    def forecast(self, prices: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
        df = prices[["date", "close"]].rename(columns={"date": "ds", "close": "y"}).copy()
        df["ds"] = pd.to_datetime(df["ds"])
        try:
            from prophet import Prophet

            model = Prophet(interval_width=self.interval_width, daily_seasonality=False)
            model.fit(df)
            future = model.make_future_dataframe(periods=horizon_days, freq="B")
            forecast = model.predict(future)
            return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)
        except Exception:
            return self._linear_fallback(df, horizon_days)

    def evaluate(self, prices: pd.DataFrame, test_days: int = 30) -> dict[str, float]:
        if len(prices) <= test_days + 20:
            raise ValueError("Not enough rows for forecast evaluation")
        train = prices.iloc[:-test_days]
        actual = prices.iloc[-test_days:]["close"].to_numpy()
        predicted = self.forecast(train, test_days)["yhat"].to_numpy()
        error = actual - predicted
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mape": float(np.mean(np.abs(error / np.maximum(actual, 1e-9))) * 100),
        }

    def _linear_fallback(self, df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        x = np.arange(len(df))
        slope, intercept = np.polyfit(x, df["y"].to_numpy(), 1)
        future_dates = pd.bdate_range(df["ds"].max() + pd.Timedelta(days=1), periods=horizon_days)
        future_x = np.arange(len(df), len(df) + horizon_days)
        yhat = slope * future_x + intercept
        residual_std = float(np.std(df["y"].to_numpy() - (slope * x + intercept)))
        return pd.DataFrame(
            {
                "ds": future_dates,
                "yhat": yhat,
                "yhat_lower": yhat - 1.28 * residual_std,
                "yhat_upper": yhat + 1.28 * residual_std,
            }
        )

