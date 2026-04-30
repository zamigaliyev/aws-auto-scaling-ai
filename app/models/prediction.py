"""
AI Load Prediction Model
Uses an ensemble of Linear Regression + Gradient Boosted Trees with
feature engineering (time-of-day, rolling averages, trend signals)
to forecast CPU/memory utilization and optimal instance counts.
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import structlog

logger = structlog.get_logger(__name__)

MODEL_PATH = "models/prediction_model.pkl"


class FeatureEngineer:
    """Transforms raw metric time-series into ML-ready feature vectors."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Temporal features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Rolling statistics (requires sorted time-series)
        for window in [3, 6, 12, 24]:
            df[f"cpu_roll_mean_{window}"] = (
                df["cpu_utilization"].rolling(window, min_periods=1).mean()
            )
            df[f"cpu_roll_std_{window}"] = (
                df["cpu_utilization"].rolling(window, min_periods=1).std().fillna(0)
            )
            df[f"mem_roll_mean_{window}"] = (
                df["memory_utilization"].rolling(window, min_periods=1).mean()
            )

        # Lag features
        for lag in [1, 2, 3, 6]:
            df[f"cpu_lag_{lag}"] = df["cpu_utilization"].shift(lag).fillna(
                df["cpu_utilization"].mean()
            )
            df[f"mem_lag_{lag}"] = df["memory_utilization"].shift(lag).fillna(
                df["memory_utilization"].mean()
            )

        # Trend (first-order difference)
        df["cpu_trend"] = df["cpu_utilization"].diff().fillna(0)
        df["mem_trend"] = df["memory_utilization"].diff().fillna(0)

        # Instance load factor
        df["load_per_instance"] = df["cpu_utilization"] / df["instance_count"].clip(lower=1)

        return df

    def get_feature_columns(self) -> list:
        return [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
            "cpu_roll_mean_3", "cpu_roll_mean_6", "cpu_roll_mean_12", "cpu_roll_mean_24",
            "cpu_roll_std_3", "cpu_roll_std_6",
            "mem_roll_mean_3", "mem_roll_mean_6", "mem_roll_mean_12",
            "cpu_lag_1", "cpu_lag_2", "cpu_lag_3", "cpu_lag_6",
            "mem_lag_1", "mem_lag_2", "mem_lag_3",
            "cpu_trend", "mem_trend",
            "load_per_instance",
            "network_in", "network_out", "request_count",
        ]


class LoadPredictionModel:
    """
    Ensemble model that predicts future CPU/memory utilization
    and recommends optimal instance counts for the next N hours.
    """

    def __init__(self, min_samples: int = 100):
        self.min_samples = min_samples
        self.fe = FeatureEngineer()
        self.cpu_model: Optional[Pipeline] = None
        self.mem_model: Optional[Pipeline] = None
        self.is_trained = False
        self.last_trained: Optional[datetime] = None
        self.training_samples = 0
        self.cpu_mae = 0.0
        self.mem_mae = 0.0
        self.r2_score = 0.0
        self._load_model()

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_split=5,
                random_state=42,
            )),
        ])

    def train(self, data_points: list) -> dict:
        df = pd.DataFrame([dp.model_dump() for dp in data_points])
        if len(df) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples, got {len(df)}"
            )

        df = self.fe.build_features(df)
        feature_cols = [c for c in self.fe.get_feature_columns() if c in df.columns]
        X = df[feature_cols].fillna(0)
        y_cpu = df["cpu_utilization"]
        y_mem = df["memory_utilization"]

        self.cpu_model = self._build_pipeline()
        self.mem_model = self._build_pipeline()
        self.cpu_model.fit(X, y_cpu)
        self.mem_model.fit(X, y_mem)

        # Cross-validated accuracy
        cpu_scores = cross_val_score(
            self._build_pipeline(), X, y_cpu, cv=3,
            scoring="neg_mean_absolute_error"
        )
        mem_scores = cross_val_score(
            self._build_pipeline(), X, y_mem, cv=3,
            scoring="neg_mean_absolute_error"
        )
        self.cpu_mae = float(-cpu_scores.mean())
        self.mem_mae = float(-mem_scores.mean())

        y_cpu_pred = self.cpu_model.predict(X)
        self.r2_score = float(r2_score(y_cpu, y_cpu_pred))

        self.is_trained = True
        self.last_trained = datetime.utcnow()
        self.training_samples = len(df)
        self._save_model()

        logger.info(
            "prediction_model_trained",
            samples=len(df),
            cpu_mae=round(self.cpu_mae, 2),
            mem_mae=round(self.mem_mae, 2),
            r2=round(self.r2_score, 4),
        )
        return {
            "samples": len(df),
            "cpu_mae": self.cpu_mae,
            "mem_mae": self.mem_mae,
            "r2_score": self.r2_score,
        }

    def predict(
        self,
        recent_data: list,
        horizon_hours: int = 2,
        include_confidence: bool = True,
    ) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet")

        df = pd.DataFrame([dp.model_dump() for dp in recent_data])
        df = self.fe.build_features(df)
        feature_cols = [c for c in self.fe.get_feature_columns() if c in df.columns]

        last_row = df.iloc[-1].copy()
        last_ts = pd.to_datetime(last_row["timestamp"])

        cpu_preds, mem_preds = [], []
        lower_cpu, upper_cpu = [], []
        lower_mem, upper_mem = [], []

        for h in range(1, horizon_hours + 1):
            future_ts = last_ts + timedelta(hours=h)
            row = last_row.copy()
            row["timestamp"] = future_ts
            row["hour"] = future_ts.hour
            row["day_of_week"] = future_ts.dayofweek
            row["is_weekend"] = int(future_ts.dayofweek >= 5)
            row["hour_sin"] = np.sin(2 * np.pi * future_ts.hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * future_ts.hour / 24)
            row["dow_sin"] = np.sin(2 * np.pi * future_ts.dayofweek / 7)
            row["dow_cos"] = np.cos(2 * np.pi * future_ts.dayofweek / 7)

            X_pred = pd.DataFrame([row[feature_cols].fillna(0)])
            cpu_p = float(np.clip(self.cpu_model.predict(X_pred)[0], 0, 100))
            mem_p = float(np.clip(self.mem_model.predict(X_pred)[0], 0, 100))
            cpu_preds.append(cpu_p)
            mem_preds.append(mem_p)

            # Confidence intervals using model MAE as proxy
            margin_cpu = self.cpu_mae * (1 + 0.1 * h)
            margin_mem = self.mem_mae * (1 + 0.1 * h)
            lower_cpu.append(max(0.0, cpu_p - margin_cpu))
            upper_cpu.append(min(100.0, cpu_p + margin_cpu))
            lower_mem.append(max(0.0, mem_p - margin_mem))
            upper_mem.append(min(100.0, mem_p + margin_mem))

            # Feed prediction back as lag for next step
            last_row["cpu_lag_2"] = last_row.get("cpu_lag_1", cpu_p)
            last_row["cpu_lag_1"] = cpu_p
            last_row["mem_lag_2"] = last_row.get("mem_lag_1", mem_p)
            last_row["mem_lag_1"] = mem_p

        instance_counts = self._cpu_to_instances(cpu_preds)

        return {
            "predicted_cpu": cpu_preds,
            "predicted_memory": mem_preds,
            "predicted_instance_count": instance_counts,
            "confidence_lower_cpu": lower_cpu if include_confidence else None,
            "confidence_upper_cpu": upper_cpu if include_confidence else None,
            "confidence_lower_mem": lower_mem if include_confidence else None,
            "confidence_upper_mem": upper_mem if include_confidence else None,
            "model_accuracy": max(0.0, 1.0 - (self.cpu_mae / 100)),
        }

    def _cpu_to_instances(self, cpu_preds: list, target_util: float = 60.0) -> list:
        """Convert predicted CPU % to recommended instance count."""
        counts = []
        for cpu in cpu_preds:
            if cpu <= 0:
                counts.append(1)
            else:
                counts.append(max(1, int(np.ceil(cpu / target_util))))
        return counts

    def get_accuracy(self) -> float:
        if not self.is_trained:
            return 0.0
        return max(0.0, 1.0 - (self.cpu_mae / 100))

    def _save_model(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(
            {"cpu": self.cpu_model, "mem": self.mem_model,
             "trained_at": self.last_trained, "samples": self.training_samples,
             "cpu_mae": self.cpu_mae, "mem_mae": self.mem_mae},
            MODEL_PATH,
        )

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.cpu_model = data["cpu"]
                self.mem_model = data["mem"]
                self.last_trained = data.get("trained_at")
                self.training_samples = data.get("samples", 0)
                self.cpu_mae = data.get("cpu_mae", 0.0)
                self.mem_mae = data.get("mem_mae", 0.0)
                self.is_trained = True
                logger.info("prediction_model_loaded", samples=self.training_samples)
            except Exception as e:
                logger.warning("prediction_model_load_failed", error=str(e))
