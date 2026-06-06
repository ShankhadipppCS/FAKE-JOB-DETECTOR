from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class MLPriceForecaster:
    def train(self, features: pd.DataFrame) -> dict[str, float]:
        df = features.dropna().copy()
        df["target_next_close"] = df["close"].shift(-1)
        df = df.dropna()
        feature_cols = [c for c in df.columns if c not in {"symbol", "date", "target_next_close"}]
        x_train, x_test, y_train, y_test = train_test_split(df[feature_cols], df["target_next_close"], test_size=0.2, shuffle=False)
        model = self._model()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        return {"mae": float(mean_absolute_error(y_test, pred)), "rmse": float(np.sqrt(np.mean((y_test - pred) ** 2)))}

    def _model(self):
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, objective="reg:squarederror")
        except Exception:
            return RandomForestRegressor(n_estimators=200, random_state=42)

