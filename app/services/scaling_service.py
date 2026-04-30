"""
Scaling Decision Engine
Combines real-time metrics, ML predictions, anomaly signals, and cost analysis
into a single ScalingDecision and executes it against the ASG.
"""

import asyncio
from datetime import datetime
from typing import Optional, List
import structlog

from app.config import Settings
from app.models.schemas import (
    ScalingDecision, ScalingDirection, ScalingTrigger, MetricDataPoint
)
from app.models.prediction import LoadPredictionModel
from app.models.anomaly_detection import AnomalyDetector
from app.services.aws_service import AWSService
from app.services.cost_service import CostService
from app.services.monitoring_service import MonitoringService

logger = structlog.get_logger(__name__)


class ScalingService:
    def __init__(
        self,
        settings: Settings,
        aws: AWSService,
        prediction_model: LoadPredictionModel,
        anomaly_detector: AnomalyDetector,
        cost_service: CostService,
        monitoring: MonitoringService,
    ):
        self.settings = settings
        self.aws = aws
        self.prediction_model = prediction_model
        self.anomaly_detector = anomaly_detector
        self.cost_service = cost_service
        self.monitoring = monitoring
        self._last_scale_time: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def evaluate_and_scale(self) -> ScalingDecision:
        """
        Main decision loop: collect metrics → detect anomalies → run ML prediction
        → apply cost check → execute scaling if needed.
        """
        asg_data = await self._get_asg()
        if asg_data is None:
            return self._no_action_decision(0, "ASG not found")

        current = asg_data["current_capacity"]
        recent_points = self.monitoring.get_recent_data_points(hours=2)

        if not recent_points:
            return self._no_action_decision(current, "Insufficient metric history")

        latest = recent_points[-1]
        cpu = latest.get("cpu_utilization", 0.0)
        mem = latest.get("memory_utilization", 0.0)

        # 1. Anomaly check
        if recent_points:
            dp = _dict_to_metric_point(latest)
            anomaly = self.anomaly_detector.detect(dp)
            if anomaly["is_anomaly"] and anomaly["severity"] in ("critical", "high"):
                logger.warning(
                    "anomaly_detected_preemptive_scale",
                    severity=anomaly["severity"],
                    score=round(anomaly["anomaly_score"], 3),
                )
                action = anomaly["recommended_action"]
                if action == "scale_out":
                    target = min(current + 2, self.settings.max_instances)
                    return await self._execute_decision(
                        current, target, ScalingTrigger.ANOMALY,
                        f"Anomaly detected (severity={anomaly['severity']})",
                        confidence=anomaly["anomaly_score"],
                    )

        # 2. Threshold-based scaling (baseline safety net)
        threshold_decision = self._threshold_decision(cpu, mem, current)
        if threshold_decision:
            return await self._execute_decision(*threshold_decision)

        # 3. ML-based predictive scaling
        if self.prediction_model.is_trained and len(recent_points) >= 10:
            metric_objs = [_dict_to_metric_point(p) for p in recent_points[-24:]]
            try:
                pred = self.prediction_model.predict(
                    metric_objs,
                    horizon_hours=self.settings.prediction_horizon_hours,
                )
                predicted_instances = pred["predicted_instance_count"]
                peak_predicted = max(predicted_instances)
                cost_ok = self.cost_service.analyze(
                    current, peak_predicted, pred["predicted_cpu"]
                )["is_cost_effective"]
                if peak_predicted != current and cost_ok:
                    direction = (
                        ScalingDirection.OUT
                        if peak_predicted > current
                        else ScalingDirection.IN
                    )
                    if self._cooldown_ok(direction):
                        return await self._execute_decision(
                            current,
                            peak_predicted,
                            ScalingTrigger.PREDICTION,
                            f"ML prediction: peak demand needs {peak_predicted} instances",
                            confidence=self.prediction_model.get_accuracy(),
                        )
            except Exception as e:
                logger.warning("ml_prediction_failed", error=str(e))

        return self._no_action_decision(current, "Metrics within normal range")

    async def scale_manually(self, target: int, reason: str) -> ScalingDecision:
        asg_data = await self._get_asg()
        current = asg_data["current_capacity"] if asg_data else 0
        return await self._execute_decision(
            current, target, ScalingTrigger.MANUAL, reason, confidence=1.0
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _threshold_decision(
        self, cpu: float, mem: float, current: int
    ) -> Optional[tuple]:
        if cpu >= self.settings.cpu_scale_out_threshold:
            target = min(current + max(1, current // 2), self.settings.max_instances)
            if self._cooldown_ok(ScalingDirection.OUT):
                return (
                    current, target, ScalingTrigger.CPU,
                    f"CPU {cpu:.1f}% ≥ threshold {self.settings.cpu_scale_out_threshold}%",
                    0.95,
                )
        elif mem >= self.settings.memory_scale_out_threshold:
            target = min(current + 1, self.settings.max_instances)
            if self._cooldown_ok(ScalingDirection.OUT):
                return (
                    current, target, ScalingTrigger.MEMORY,
                    f"Memory {mem:.1f}% ≥ threshold {self.settings.memory_scale_out_threshold}%",
                    0.90,
                )
        elif (
            cpu <= self.settings.cpu_scale_in_threshold
            and current > self.settings.min_instances
        ):
            target = max(current - 1, self.settings.min_instances)
            if self._cooldown_ok(ScalingDirection.IN):
                return (
                    current, target, ScalingTrigger.CPU,
                    f"CPU {cpu:.1f}% ≤ scale-in threshold {self.settings.cpu_scale_in_threshold}%",
                    0.85,
                )
        return None

    async def _execute_decision(
        self,
        current: int,
        target: int,
        trigger: ScalingTrigger,
        reason: str,
        confidence: float = 0.9,
    ) -> ScalingDecision:
        if target == current:
            return self._no_action_decision(current, "Target equals current capacity")

        direction = ScalingDirection.OUT if target > current else ScalingDirection.IN
        cost_delta = (target - current) * self.settings.cost_per_instance_hour

        decision = ScalingDecision(
            direction=direction,
            current_capacity=current,
            target_capacity=target,
            trigger=trigger,
            confidence=min(1.0, max(0.0, confidence)),
            reason=reason,
            estimated_cost_delta=cost_delta,
        )

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.aws.set_desired_capacity(target)
            )
            self._last_scale_time = datetime.utcnow()
            self.cost_service.record_scaling_action(
                current, target, trigger.value, 0.0
            )
            logger.info(
                "scaling_executed",
                direction=direction.value,
                from_count=current,
                to_count=target,
                trigger=trigger.value,
                confidence=round(confidence, 3),
            )
        except Exception as e:
            logger.error("scaling_execution_failed", error=str(e))
            decision.reason = f"FAILED: {reason} — {e}"

        return decision

    def _cooldown_ok(self, direction: ScalingDirection) -> bool:
        if self._last_scale_time is None:
            return True
        from datetime import timedelta
        elapsed = (datetime.utcnow() - self._last_scale_time).total_seconds()
        cooldown = (
            self.settings.scale_out_cooldown_seconds
            if direction == ScalingDirection.OUT
            else self.settings.scale_in_cooldown_seconds
        )
        if elapsed < cooldown:
            logger.debug("scaling_in_cooldown", elapsed=elapsed, cooldown=cooldown)
            return False
        return True

    async def _get_asg(self) -> Optional[dict]:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.aws.describe_asg)
        except Exception as e:
            logger.error("asg_fetch_failed", error=str(e))
            return None

    @staticmethod
    def _no_action_decision(current: int, reason: str) -> ScalingDecision:
        return ScalingDecision(
            direction=ScalingDirection.NONE,
            current_capacity=current,
            target_capacity=current,
            trigger=ScalingTrigger.CPU,
            confidence=1.0,
            reason=reason,
            estimated_cost_delta=0.0,
        )


def _dict_to_metric_point(d: dict) -> MetricDataPoint:
    return MetricDataPoint(
        timestamp=d.get("timestamp", datetime.utcnow()),
        cpu_utilization=float(d.get("cpu_utilization", 0.0)),
        memory_utilization=float(d.get("memory_utilization", 0.0)),
        network_in=float(d.get("network_in", 0.0)),
        network_out=float(d.get("network_out", 0.0)),
        request_count=int(d.get("request_count", 0)),
        instance_count=max(1, int(d.get("instance_count", 1))),
    )
