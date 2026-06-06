from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalIndicatorEngine:
    def add_all(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices.copy().sort_values("date")
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["rsi_14"] = self.rsi(df["close"], 14)
        stoch = self.stochastic(df, 14)
        df["stoch_k"] = stoch["stoch_k"]
        df["stoch_d"] = stoch["stoch_d"]
        bands = self.bollinger_bands(df["close"], 20)
        df = pd.concat([df, bands], axis=1)
        df["atr_14"] = self.atr(df, 14)
        df["obv"] = self.obv(df)
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].replace(0, np.nan).cumsum()
        return df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def stochastic(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        low_min = df["low"].rolling(window).min()
        high_max = df["high"].rolling(window).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        return pd.DataFrame({"stoch_k": k.fillna(50), "stoch_d": k.rolling(3).mean().fillna(50)})

    @staticmethod
    def bollinger_bands(close: pd.Series, window: int = 20, deviations: float = 2.0) -> pd.DataFrame:
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()
        return pd.DataFrame(
            {
                "bb_middle": middle,
                "bb_upper": middle + deviations * std,
                "bb_lower": middle - deviations * std,
            }
        )

    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        prev_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(window).mean()

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        direction = np.sign(df["close"].diff()).fillna(0)
        return (direction * df["volume"]).cumsum()

