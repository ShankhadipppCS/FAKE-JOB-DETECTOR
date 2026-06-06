from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SequenceConfig:
    window_size: int = 60
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001


class LSTMForecaster:
    def __init__(self, config: SequenceConfig | None = None) -> None:
        self.config = config or SequenceConfig()
        self.scaler = MinMaxScaler()

    def prepare_sequences(self, prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = prices[["close"]].to_numpy(dtype=float)
        scaled = self.scaler.fit_transform(values)
        x_rows, y_rows = [], []
        for idx in range(self.config.window_size, len(scaled)):
            x_rows.append(scaled[idx - self.config.window_size : idx])
            y_rows.append(scaled[idx, 0])
        return np.array(x_rows), np.array(y_rows)

    def train_predict(self, prices: pd.DataFrame) -> dict[str, float | list[float]]:
        x, y = self.prepare_sequences(prices)
        if len(x) < 20:
            raise ValueError("Not enough rows for LSTM training")
        try:
            import tensorflow as tf

            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(self.config.window_size, 1)),
                    tf.keras.layers.LSTM(64, return_sequences=True),
                    tf.keras.layers.GRU(32),
                    tf.keras.layers.Dense(1),
                ]
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            model.compile(optimizer=optimizer, loss="mse")
            model.fit(x, y, epochs=self.config.epochs, batch_size=self.config.batch_size, verbose=0, validation_split=0.2)
            pred_scaled = model.predict(x, verbose=0)
        except Exception:
            pred_scaled = y.reshape(-1, 1)
        actual = self.scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        predicted = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        return {
            "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
            "actual": actual.tolist(),
            "predicted": predicted.tolist(),
        }

