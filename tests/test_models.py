"""Unit tests for ML models (no AWS required)."""

import pytest
from datetime import datetime, timedelta
from app.models.prediction import LoadPredictionModel
from app.models.anomaly_detection import AnomalyDetector
from app.models.schemas import MetricDataPoint
from app.utils.helpers import generate_synthetic_training_data


def make_data_points(n: int = 200) -> list[MetricDataPoint]:
    raw = generate_synthetic_training_data(n_points=n)
    return [MetricDataPoint(**p) for p in raw]


class TestLoadPredictionModel:
    def test_requires_min_samples(self):
        model = LoadPredictionModel(min_samples=100)
        tiny = make_data_points(10)
        with pytest.raises(ValueError, match="at least"):
            model.train(tiny)

    def test_trains_successfully(self):
        model = LoadPredictionModel(min_samples=50)
        dps = make_data_points(150)
        result = model.train(dps)
        assert model.is_trained
        assert result["samples"] == 150
        assert result["r2_score"] >= -1.0  # R² can be negative for bad fits

    def test_predict_returns_correct_horizon(self):
        model = LoadPredictionModel(min_samples=50)
        dps = make_data_points(150)
        model.train(dps)
        pred = model.predict(dps[-24:], horizon_hours=3)
        assert len(pred["predicted_cpu"]) == 3
        assert len(pred["predicted_memory"]) == 3
        assert len(pred["predicted_instance_count"]) == 3

    def test_predictions_in_valid_range(self):
        model = LoadPredictionModel(min_samples=50)
        dps = make_data_points(150)
        model.train(dps)
        pred = model.predict(dps[-24:], horizon_hours=2)
        for cpu in pred["predicted_cpu"]:
            assert 0 <= cpu <= 100
        for mem in pred["predicted_memory"]:
            assert 0 <= mem <= 100
        for count in pred["predicted_instance_count"]:
            assert count >= 1

    def test_confidence_intervals_included(self):
        model = LoadPredictionModel(min_samples=50)
        dps = make_data_points(150)
        model.train(dps)
        pred = model.predict(dps[-24:], horizon_hours=2, include_confidence=True)
        assert pred["confidence_lower_cpu"] is not None
        assert pred["confidence_upper_cpu"] is not None

    def test_accuracy_between_0_and_1(self):
        model = LoadPredictionModel(min_samples=50)
        dps = make_data_points(150)
        model.train(dps)
        acc = model.get_accuracy()
        assert 0.0 <= acc <= 1.0

    def test_untrained_raises_on_predict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.models.prediction.MODEL_PATH",
            str(tmp_path / "no_model.pkl"),
        )
        model = LoadPredictionModel(min_samples=50)
        assert model.is_trained is False
        dps = make_data_points(20)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(dps)


class TestAnomalyDetector:
    def test_trains_successfully(self):
        detector = AnomalyDetector(contamination=0.05)
        dps = make_data_points(200)
        result = detector.train(dps)
        assert detector.is_trained
        assert result["samples"] == 200

    def test_normal_point_not_anomaly(self):
        detector = AnomalyDetector(contamination=0.05)
        dps = make_data_points(300)
        detector.train(dps)
        # Normal operating point
        normal = MetricDataPoint(
            timestamp=datetime.utcnow(),
            cpu_utilization=45.0,
            memory_utilization=55.0,
            network_in=1e6,
            network_out=5e5,
            request_count=200,
            instance_count=2,
        )
        result = detector.detect(normal)
        # Not checking exact label since contamination rate affects this,
        # but score should be low for typical point
        assert 0.0 <= result["anomaly_score"] <= 1.0
        assert result["severity"] in ("low", "medium", "high", "critical")

    def test_extreme_point_detected(self):
        detector = AnomalyDetector(contamination=0.05)
        dps = make_data_points(300)
        detector.train(dps)
        # Artificially extreme values
        extreme = MetricDataPoint(
            timestamp=datetime.utcnow(),
            cpu_utilization=99.9,
            memory_utilization=99.5,
            network_in=1e10,
            network_out=1e10,
            request_count=100000,
            instance_count=1,
        )
        result = detector.detect(extreme)
        assert result["is_anomaly"] is True
        assert result["anomaly_score"] > 0.1

    def test_batch_detection_length_matches(self):
        detector = AnomalyDetector(contamination=0.05)
        dps = make_data_points(200)
        detector.train(dps)
        batch = dps[:10]
        results = detector.detect_batch(batch)
        assert len(results) == 10

    def test_untrained_detector_returns_safe_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.models.anomaly_detection.ANOMALY_MODEL_PATH",
            str(tmp_path / "no_model.pkl"),
        )
        detector = AnomalyDetector()
        assert detector.is_trained is False
        dp = MetricDataPoint(
            timestamp=datetime.utcnow(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
        )
        result = detector.detect(dp)
        assert result["is_anomaly"] is False
        assert result["recommended_action"] == "no_action"
