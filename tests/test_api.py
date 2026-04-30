"""
Integration tests for FastAPI endpoints.
AWS calls are mocked via pytest-mock so no real AWS account is needed.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import create_app
from app.config import Settings
from app.models.prediction import LoadPredictionModel
from app.models.anomaly_detection import AnomalyDetector
from app.services.aws_service import AWSService
from app.services.cost_service import CostService
from app.services.monitoring_service import MonitoringService
from app.services.scaling_service import ScalingService
from app.core.feedback_loop import FeedbackLoop
from app.core.scheduler import BackgroundScheduler
from app.models.schemas import ScalingDirection, ScalingDecision, ScalingTrigger
from app.utils.helpers import generate_synthetic_training_data


@pytest.fixture
def mock_settings():
    return Settings(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        aws_region="us-east-1",
        auto_scaling_group_name="test-asg",
        min_instances=1,
        max_instances=10,
        desired_capacity=2,
        min_training_samples=50,
    )


@pytest.fixture
def mock_asg_data():
    return {
        "group_name": "test-asg",
        "min_size": 1,
        "max_size": 10,
        "desired_capacity": 2,
        "current_capacity": 2,
        "instances": [
            {
                "instance_id": "i-abc123",
                "state": "InService",
                "instance_type": "t3.medium",
                "launch_time": None,
                "private_ip": "10.0.0.1",
                "public_ip": None,
                "availability_zone": "us-east-1a",
            }
        ],
        "health_check_type": "EC2",
        "created_time": None,
    }


@pytest.fixture
def client(mock_settings, mock_asg_data):
    app = create_app()

    aws = MagicMock(spec=AWSService)
    aws.describe_asg.return_value = mock_asg_data
    aws.set_desired_capacity.return_value = {"new_desired_capacity": 3}
    aws.describe_alarms.return_value = []
    aws.get_asg_cpu_metrics.return_value = []
    aws.check_connectivity = AsyncMock(return_value=True)

    prediction_model = MagicMock(spec=LoadPredictionModel)
    prediction_model.is_trained = False
    prediction_model.last_trained = None
    prediction_model.training_samples = 0
    prediction_model.get_accuracy.return_value = 0.0
    prediction_model.cpu_mae = 0.0
    prediction_model.mem_mae = 0.0
    prediction_model.r2_score = 0.0

    anomaly_detector = MagicMock(spec=AnomalyDetector)
    anomaly_detector.is_trained = False
    anomaly_detector.last_trained = None
    anomaly_detector.training_samples = 0
    anomaly_detector.contamination = 0.05

    cost_service = CostService(mock_settings)
    monitoring = MagicMock(spec=MonitoringService)
    monitoring.get_recent_data_points.return_value = []
    monitoring.get_system_status = AsyncMock(return_value={
        "timestamp": datetime.utcnow(),
        "asg_status": mock_asg_data,
        "metrics": [],
        "active_alarms": [],
        "system_healthy": True,
    })

    scaling = MagicMock(spec=ScalingService)
    no_action = ScalingDecision(
        direction=ScalingDirection.NONE,
        current_capacity=2,
        target_capacity=2,
        trigger=ScalingTrigger.CPU,
        confidence=1.0,
        reason="Within normal range",
        estimated_cost_delta=0.0,
    )
    scaling.evaluate_and_scale = AsyncMock(return_value=no_action)
    scaling.scale_manually = AsyncMock(return_value=ScalingDecision(
        direction=ScalingDirection.OUT,
        current_capacity=2,
        target_capacity=3,
        trigger=ScalingTrigger.MANUAL,
        confidence=1.0,
        reason="Manual",
        estimated_cost_delta=0.096,
    ))

    feedback_loop = MagicMock(spec=FeedbackLoop)
    feedback_loop.get_next_retrain_time.return_value = None

    scheduler = MagicMock(spec=BackgroundScheduler)
    scheduler.uptime_seconds = 42.0

    # Patch lifespan so we control state
    with patch("app.main.get_settings", return_value=mock_settings), \
         patch("app.main.AWSService", return_value=aws), \
         patch("app.main.LoadPredictionModel", return_value=prediction_model), \
         patch("app.main.AnomalyDetector", return_value=anomaly_detector), \
         patch("app.main.CostService", return_value=cost_service), \
         patch("app.main.MonitoringService", return_value=monitoring), \
         patch("app.main.ScalingService", return_value=scaling), \
         patch("app.main.FeedbackLoop", return_value=feedback_loop), \
         patch("app.main.BackgroundScheduler", return_value=scheduler):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_ping(self, client):
        r = client.get("/api/v1/health/ping")
        assert r.status_code == 200
        assert r.json()["pong"] is True

    def test_health_check(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["aws_connected"] is True
        assert data["version"] == "1.0.0"
        assert "model_status" in data


class TestScalingEndpoints:
    def test_get_asg_status(self, client):
        r = client.get("/api/v1/scaling/status")
        assert r.status_code == 200
        data = r.json()
        assert data["group_name"] == "test-asg"
        assert data["current_capacity"] == 2

    def test_manual_scale(self, client):
        r = client.post("/api/v1/scaling/scale", json={"desired_capacity": 3})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["decision"]["target_capacity"] == 3

    def test_manual_scale_exceeds_max(self, client):
        r = client.post("/api/v1/scaling/scale", json={"desired_capacity": 999})
        assert r.status_code == 400

    def test_evaluate_endpoint(self, client):
        r = client.post("/api/v1/scaling/evaluate")
        assert r.status_code == 200

    def test_cost_analysis(self, client):
        r = client.get("/api/v1/scaling/cost")
        assert r.status_code == 200
        data = r.json()
        assert "current_hourly_cost" in data
        assert "optimization_suggestions" in data


class TestMonitoringEndpoints:
    def test_recent_metrics_empty(self, client):
        r = client.get("/api/v1/monitoring/metrics/recent?hours=1")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 0

    def test_ingest_metric(self, client):
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_utilization": 55.0,
            "memory_utilization": 60.0,
            "network_in": 1000000.0,
            "network_out": 500000.0,
            "request_count": 300,
            "instance_count": 2,
        }
        r = client.post("/api/v1/monitoring/metrics/ingest", json=payload)
        assert r.status_code == 200
        assert r.json()["ingested"] is True

    def test_list_alarms(self, client):
        r = client.get("/api/v1/monitoring/alarms")
        assert r.status_code == 200
        assert r.json()["count"] == 0


class TestPredictionEndpoints:
    def test_predict_without_training_returns_422(self, client):
        r = client.post("/api/v1/prediction/predict", json={"horizon_hours": 2})
        assert r.status_code == 422

    def test_model_status_untrained(self, client):
        r = client.get("/api/v1/prediction/model/status")
        assert r.status_code == 200
        data = r.json()
        assert data["prediction_model"]["trained"] is False

    def test_synthetic_train_starts(self, client):
        r = client.post("/api/v1/prediction/train/synthetic?n_points=100")
        assert r.status_code == 200
        assert "started" in r.json()["message"].lower()

    def test_detect_anomaly_without_model_raises(self, client):
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_utilization": 95.0,
            "memory_utilization": 90.0,
            "network_in": 1e9,
            "network_out": 1e9,
            "request_count": 50000,
            "instance_count": 1,
        }
        r = client.post("/api/v1/prediction/anomaly/detect", json=payload)
        assert r.status_code == 422
