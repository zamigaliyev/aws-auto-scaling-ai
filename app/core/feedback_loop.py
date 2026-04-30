"""
Feedback Loop & Continuous Learning
Periodically retrains the ML models using the latest buffered observations,
keeping prediction accuracy aligned with evolving workload patterns.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
import structlog

from app.config import Settings
from app.models.prediction import LoadPredictionModel
from app.models.anomaly_detection import AnomalyDetector
from app.models.schemas import MetricDataPoint
from app.services.monitoring_service import MonitoringService

logger = structlog.get_logger(__name__)


class FeedbackLoop:
    def __init__(
        self,
        settings: Settings,
        prediction_model: LoadPredictionModel,
        anomaly_detector: AnomalyDetector,
        monitoring: MonitoringService,
    ):
        self.settings = settings
        self.prediction_model = prediction_model
        self.anomaly_detector = anomaly_detector
        self.monitoring = monitoring
        self._last_retrain: Optional[datetime] = None
        self._retrain_count = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("feedback_loop_started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("feedback_loop_stopped")

    async def _loop(self):
        interval = self.settings.model_retrain_interval_hours * 3600
        while self._running:
            await asyncio.sleep(interval)
            await self.retrain_if_due()

    async def retrain_if_due(self) -> dict:
        """Retrain models if interval has elapsed and enough data is available."""
        if self._last_retrain is not None:
            hours_since = (
                datetime.utcnow() - self._last_retrain
            ).total_seconds() / 3600
            if hours_since < self.settings.model_retrain_interval_hours:
                logger.debug(
                    "retrain_not_due",
                    hours_until_next=round(
                        self.settings.model_retrain_interval_hours - hours_since, 2
                    ),
                )
                return {"skipped": True, "reason": "retrain_not_due"}

        all_points = self.monitoring.buffer.all()
        if len(all_points) < self.settings.min_training_samples:
            logger.info(
                "retrain_skipped_insufficient_data",
                have=len(all_points),
                need=self.settings.min_training_samples,
            )
            return {
                "skipped": True,
                "reason": "insufficient_data",
                "samples": len(all_points),
            }

        metric_objs = [_dict_to_metric_point(p) for p in all_points]
        result = {"timestamp": datetime.utcnow().isoformat()}

        # Retrain prediction model
        try:
            pred_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.prediction_model.train(metric_objs)
            )
            result["prediction_model"] = pred_result
            logger.info("prediction_model_retrained", **pred_result)
        except Exception as e:
            logger.error("prediction_model_retrain_failed", error=str(e))
            result["prediction_model_error"] = str(e)

        # Retrain anomaly detector
        try:
            anom_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.anomaly_detector.train(metric_objs)
            )
            result["anomaly_model"] = anom_result
            logger.info("anomaly_model_retrained", **anom_result)
        except Exception as e:
            logger.error("anomaly_model_retrain_failed", error=str(e))
            result["anomaly_model_error"] = str(e)

        self._last_retrain = datetime.utcnow()
        self._retrain_count += 1
        result["retrain_count"] = self._retrain_count
        return result

    def get_next_retrain_time(self) -> Optional[datetime]:
        if self._last_retrain is None:
            return None
        return self._last_retrain + timedelta(
            hours=self.settings.model_retrain_interval_hours
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
