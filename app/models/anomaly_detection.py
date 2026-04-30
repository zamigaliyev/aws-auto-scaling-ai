"""
Anomaly Detection Engine
Uses Isolation Forest (unsupervised) to detect abnormal resource usage patterns
and trigger preemptive scaling before thresholds are breached.
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from typing import Optional, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import structlog

logger = structlog.get_logger(__name__)

ANOMALY_MODEL_PATH = "models/anomaly_model.pkl"


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector.
    Learns the normal operational envelope of the cluster and flags
    deviations that warrant preemptive action.
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model: Optional[Pipeline] = None
        self.is_trained = False
        self.last_trained: Optional[datetime] = None
        self.training_samples = 0
        self.feature_names: List[str] = []
        self._load_model()

    FEATURE_COLUMNS = [
        "cpu_utilization",
        "memory_utilization",
        "network_in",
        "network_out",
        "request_count",
        "cpu_mem_ratio",
        "cpu_change_rate",
        "mem_change_rate",
        "load_per_instance",
    ]

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cpu_mem_ratio"] = df["cpu_utilization"] / (df["memory_utilization"].clip(lower=1))
        df["cpu_change_rate"] = df["cpu_utilization"].diff().fillna(0).abs()
        df["mem_change_rate"] = df["memory_utilization"].diff().fillna(0).abs()
        df["load_per_instance"] = df["cpu_utilization"] / df["instance_count"].clip(lower=1)
        return df

    def train(self, data_points: list) -> dict:
        df = pd.DataFrame([dp.model_dump() for dp in data_points])
        df = self._build_features(df)

        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        X = df[available].fillna(0)
        self.feature_names = available

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("detector", IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                max_samples="auto",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self.model.fit(X)
        self.is_trained = True
        self.last_trained = datetime.utcnow()
        self.training_samples = len(df)
        self._save_model()

        # Estimate training anomaly rate
        labels = self.model.predict(X)
        anomaly_rate = float((labels == -1).sum() / len(labels))
        logger.info(
            "anomaly_model_trained",
            samples=len(df),
            detected_anomaly_rate=round(anomaly_rate, 4),
        )
        return {"samples": len(df), "anomaly_rate_in_training": anomaly_rate}

    def detect(self, data_point) -> dict:
        if not self.is_trained:
            return self._no_model_result()

        df = pd.DataFrame([data_point.model_dump()])
        df = self._build_features(df)
        available = [c for c in self.feature_names if c in df.columns]
        X = df[available].fillna(0)

        # IsolationForest: -1 = anomaly, +1 = normal
        label = int(self.model.predict(X)[0])
        score = float(self.model.named_steps["detector"].score_samples(
            self.model.named_steps["scaler"].transform(X)
        )[0])

        is_anomaly = label == -1
        # Normalize score to [0, 1] where 1 = most anomalous
        anomaly_score = float(np.clip(-score, 0, 1))

        affected = self._identify_affected_metrics(data_point, anomaly_score)
        severity = self._classify_severity(anomaly_score)

        logger.debug(
            "anomaly_detection_result",
            is_anomaly=is_anomaly,
            score=round(anomaly_score, 4),
            severity=severity,
        )
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "affected_metrics": affected,
            "severity": severity,
            "recommended_action": self._recommend_action(
                is_anomaly, anomaly_score, data_point
            ),
        }

    def detect_batch(self, data_points: list) -> List[dict]:
        """Score a batch for time-series anomaly analysis."""
        if not self.is_trained:
            return [self._no_model_result() for _ in data_points]

        df = pd.DataFrame([dp.model_dump() for dp in data_points])
        df = self._build_features(df)
        available = [c for c in self.feature_names if c in df.columns]
        X = df[available].fillna(0)

        labels = self.model.predict(X)
        scores = self.model.named_steps["detector"].score_samples(
            self.model.named_steps["scaler"].transform(X)
        )

        results = []
        for i, (dp, label, score) in enumerate(zip(data_points, labels, scores)):
            is_anomaly = label == -1
            anomaly_score = float(np.clip(-score, 0, 1))
            results.append({
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_score,
                "affected_metrics": self._identify_affected_metrics(dp, anomaly_score),
                "severity": self._classify_severity(anomaly_score),
                "recommended_action": self._recommend_action(
                    is_anomaly, anomaly_score, dp
                ),
            })
        return results

    def _identify_affected_metrics(self, dp, score: float) -> List[str]:
        affected = []
        if hasattr(dp, "cpu_utilization") and dp.cpu_utilization > 75:
            affected.append("cpu_utilization")
        if hasattr(dp, "memory_utilization") and dp.memory_utilization > 80:
            affected.append("memory_utilization")
        if hasattr(dp, "network_in") and dp.network_in > 1e8:
            affected.append("network_in")
        if hasattr(dp, "network_out") and dp.network_out > 1e8:
            affected.append("network_out")
        if not affected and score > 0.3:
            affected.append("composite_pattern")
        return affected

    def _classify_severity(self, score: float) -> str:
        if score > 0.7:
            return "critical"
        elif score > 0.5:
            return "high"
        elif score > 0.3:
            return "medium"
        return "low"

    def _recommend_action(self, is_anomaly: bool, score: float, dp) -> str:
        if not is_anomaly:
            return "no_action"
        cpu = getattr(dp, "cpu_utilization", 0)
        mem = getattr(dp, "memory_utilization", 0)
        if cpu > 70 or mem > 80:
            return "scale_out"
        if cpu < 15 and mem < 20:
            return "scale_in"
        return "scale_out"  # Default: protect availability

    def _no_model_result(self) -> dict:
        return {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "affected_metrics": [],
            "severity": "unknown",
            "recommended_action": "no_action",
        }

    def _save_model(self):
        os.makedirs(os.path.dirname(ANOMALY_MODEL_PATH), exist_ok=True)
        joblib.dump(
            {"model": self.model, "features": self.feature_names,
             "trained_at": self.last_trained, "samples": self.training_samples},
            ANOMALY_MODEL_PATH,
        )

    def _load_model(self):
        if os.path.exists(ANOMALY_MODEL_PATH):
            try:
                data = joblib.load(ANOMALY_MODEL_PATH)
                self.model = data["model"]
                self.feature_names = data.get("features", self.FEATURE_COLUMNS)
                self.last_trained = data.get("trained_at")
                self.training_samples = data.get("samples", 0)
                self.is_trained = True
                logger.info("anomaly_model_loaded", samples=self.training_samples)
            except Exception as e:
                logger.warning("anomaly_model_load_failed", error=str(e))
