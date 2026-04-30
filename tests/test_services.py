"""Unit tests for service layer (Cost, Monitoring buffer, ScalingService logic)."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

from app.config import Settings
from app.services.cost_service import CostService
from app.services.monitoring_service import MonitoringService, MetricBuffer, _trend


@pytest.fixture
def settings():
    return Settings(
        cost_per_instance_hour=0.096,
        min_instances=1,
        max_instances=20,
        cpu_scale_out_threshold=80.0,
        cpu_scale_in_threshold=20.0,
    )


class TestCostService:
    def test_current_hourly_cost(self, settings):
        svc = CostService(settings)
        result = svc.analyze(4, 4, [50.0] * 5)
        assert result["current_hourly_cost"] == pytest.approx(4 * 0.096, rel=1e-3)

    def test_scale_out_positive_delta(self, settings):
        svc = CostService(settings)
        result = svc.analyze(2, 4, [75.0] * 5)
        assert result["delta_hourly"] > 0

    def test_scale_in_negative_delta(self, settings):
        svc = CostService(settings)
        result = svc.analyze(4, 2, [15.0] * 5)
        assert result["delta_hourly"] < 0

    def test_scale_out_cost_effective_at_high_cpu(self, settings):
        svc = CostService(settings)
        result = svc.analyze(2, 4, [85.0, 88.0, 90.0])
        assert result["is_cost_effective"] is True

    def test_scale_in_cost_effective_at_low_cpu(self, settings):
        svc = CostService(settings)
        result = svc.analyze(4, 2, [10.0, 12.0, 8.0])
        assert result["is_cost_effective"] is True

    def test_suggestions_not_empty(self, settings):
        svc = CostService(settings)
        result = svc.analyze(2, 2, [50.0])
        assert len(result["optimization_suggestions"]) >= 1

    def test_scaling_history_recorded(self, settings):
        svc = CostService(settings)
        svc.record_scaling_action(2, 4, "cpu", 82.0)
        svc.record_scaling_action(4, 2, "cpu", 15.0)
        report = svc.get_scaling_cost_report(hours=1)
        assert report["total_scaling_actions"] == 2
        assert report["scale_out_actions"] == 1
        assert report["scale_in_actions"] == 1

    def test_thrash_detection(self, settings):
        svc = CostService(settings)
        for _ in range(3):
            svc.record_scaling_action(2, 4, "cpu", 82.0)
            svc.record_scaling_action(4, 2, "cpu", 18.0)
        report = svc.get_scaling_cost_report(hours=1)
        assert report["unnecessary_actions"] >= 1


class TestMetricBuffer:
    def test_push_and_retrieve(self):
        buf = MetricBuffer(maxlen=10)
        buf.push({"cpu": 50})
        assert len(buf) == 1
        assert buf.last(1)[0]["cpu"] == 50

    def test_maxlen_enforced(self):
        buf = MetricBuffer(maxlen=5)
        for i in range(10):
            buf.push({"i": i})
        assert len(buf) == 5
        assert buf.last(1)[0]["i"] == 9

    def test_last_n(self):
        buf = MetricBuffer()
        for i in range(20):
            buf.push({"i": i})
        last5 = buf.last(5)
        assert len(last5) == 5
        assert last5[-1]["i"] == 19


class TestTrendHelper:
    def _make_buf(self, values, metric="cpu_utilization"):
        return [{metric: v} for v in values]

    def test_rising_trend(self):
        buf = self._make_buf([10, 20, 30, 40, 50, 60, 70, 80])
        assert _trend(buf, "cpu_utilization") == "rising"

    def test_falling_trend(self):
        buf = self._make_buf([80, 70, 60, 50, 40, 30, 20, 10])
        assert _trend(buf, "cpu_utilization") == "falling"

    def test_stable_trend(self):
        buf = self._make_buf([50, 51, 49, 50, 51, 50, 49, 50])
        assert _trend(buf, "cpu_utilization") == "stable"

    def test_too_short_returns_stable(self):
        buf = self._make_buf([50, 60])
        assert _trend(buf, "cpu_utilization") == "stable"
