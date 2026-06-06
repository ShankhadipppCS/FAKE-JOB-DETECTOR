from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib

from app.config.settings import settings
from app.forecasting.ml_models import MLPriceForecaster


class RetrainingPipeline:
    def __init__(self, registry_dir: str = settings.model_registry_dir) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def train_feature_model(self, features):
        model = MLPriceForecaster()
        metrics = model.train(features)
        version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        artifact = self.registry_dir / f"xgboost_or_rf_{version}.joblib"
        joblib.dump({"metrics": metrics, "created_at": version}, artifact)
        self._log_mlflow(metrics)
        return {"version": version, "artifact": str(artifact), "metrics": metrics}

    def _log_mlflow(self, metrics: dict[str, float]) -> None:
        try:
            import mlflow

            with mlflow.start_run(run_name="stock-price-retraining"):
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
        except Exception:
            return

