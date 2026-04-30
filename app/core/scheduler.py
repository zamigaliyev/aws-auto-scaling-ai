"""
Background Scheduler
Drives the polling loop: every 30 seconds it collects metrics, runs anomaly
detection, evaluates scaling, and publishes custom CloudWatch metrics.
"""

import asyncio
from datetime import datetime
from typing import Optional
import structlog

from app.config import Settings
from app.services.monitoring_service import MonitoringService
from app.services.scaling_service import ScalingService
from app.models.anomaly_detection import AnomalyDetector
from app.models.schemas import MetricDataPoint

logger = structlog.get_logger(__name__)

POLL_INTERVAL_SECONDS = 30


class BackgroundScheduler:
    def __init__(
        self,
        settings: Settings,
        monitoring: MonitoringService,
        scaling: ScalingService,
        anomaly_detector: AnomalyDetector,
    ):
        self.settings = settings
        self.monitoring = monitoring
        self.scaling = scaling
        self.anomaly_detector = anomaly_detector
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._iteration = 0
        self.start_time = datetime.utcnow()

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("background_scheduler_started", interval_s=POLL_INTERVAL_SECONDS)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("background_scheduler_stopped", iterations=self._iteration)

    async def _run(self):
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("scheduler_tick_error", error=str(e), iteration=self._iteration)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _tick(self):
        self._iteration += 1

        # 1. Collect metrics from CloudWatch
        point = await self.monitoring.collect_metrics()
        if not point:
            logger.debug("scheduler_no_metrics", iteration=self._iteration)
            return

        # 2. Real-time anomaly check on latest point
        if self.anomaly_detector.is_trained:
            dp = MetricDataPoint(
                timestamp=point.get("timestamp", datetime.utcnow()),
                cpu_utilization=float(point.get("cpu_utilization", 0.0)),
                memory_utilization=float(point.get("memory_utilization", 0.0)),
                network_in=float(point.get("network_in", 0.0)),
                network_out=float(point.get("network_out", 0.0)),
                request_count=int(point.get("request_count", 0)),
                instance_count=max(1, int(point.get("instance_count", 1))),
            )
            anomaly = self.anomaly_detector.detect(dp)
            if anomaly["is_anomaly"]:
                logger.warning(
                    "real_time_anomaly",
                    severity=anomaly["severity"],
                    score=round(anomaly["anomaly_score"], 3),
                    metrics=anomaly["affected_metrics"],
                )
                # Publish anomaly score to CloudWatch for dashboards
                self.monitoring.publish_custom_metric(
                    "AnomalyScore", anomaly["anomaly_score"] * 100, "Percent"
                )

        # 3. Evaluate scaling every 5th iteration (2.5 min cadence)
        if self._iteration % 5 == 0:
            decision = await self.scaling.evaluate_and_scale()
            logger.info(
                "auto_scale_evaluation",
                direction=decision.direction.value,
                current=decision.current_capacity,
                target=decision.target_capacity,
                trigger=decision.trigger.value,
            )
            # Publish scaling signal to CloudWatch
            self.monitoring.publish_custom_metric(
                "DesiredCapacity", float(decision.target_capacity), "Count"
            )

    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()
