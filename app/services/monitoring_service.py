"""
Monitoring Service
Pulls real-time CloudWatch metrics, aggregates them into a system health
snapshot, and maintains an in-memory circular buffer of recent observations.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Deque
import structlog

from app.config import Settings
from app.services.aws_service import AWSService

logger = structlog.get_logger(__name__)

MAX_BUFFER_SIZE = 2880  # 24 h @ 30-second intervals


class MetricBuffer:
    """Thread-safe circular buffer for recent metric observations."""

    def __init__(self, maxlen: int = MAX_BUFFER_SIZE):
        self._buf: Deque[Dict] = deque(maxlen=maxlen)

    def push(self, point: Dict):
        self._buf.append(point)

    def last(self, n: int = 1) -> List[Dict]:
        items = list(self._buf)
        return items[-n:] if n <= len(items) else items

    def all(self) -> List[Dict]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


class MonitoringService:
    def __init__(self, settings: Settings, aws: AWSService):
        self.settings = settings
        self.aws = aws
        self.buffer = MetricBuffer()
        self._last_poll: Optional[datetime] = None

    async def collect_metrics(self) -> Dict:
        """Pull current metrics from CloudWatch and push to the buffer."""
        loop = asyncio.get_event_loop()
        try:
            datapoints = await loop.run_in_executor(
                None, lambda: self.aws.get_asg_cpu_metrics(hours=1)
            )
            if not datapoints:
                return {}

            latest = datapoints[-1]
            cpu = latest.get("Average", 0.0)
            point = {
                "timestamp": datetime.utcnow(),
                "cpu_utilization": cpu,
                "memory_utilization": 0.0,  # CloudWatch agent required for real data
                "network_in": 0.0,
                "network_out": 0.0,
                "request_count": 0,
                "instance_count": await self._get_instance_count(),
            }
            self.buffer.push(point)
            self._last_poll = datetime.utcnow()
            return point
        except Exception as e:
            logger.error("metric_collection_failed", error=str(e))
            return {}

    def ingest_metric(self, point: Dict):
        """Accept a metric pushed directly (e.g., from agent or test harness)."""
        if "timestamp" not in point:
            point["timestamp"] = datetime.utcnow()
        self.buffer.push(point)

    async def get_system_status(self) -> Dict:
        loop = asyncio.get_event_loop()
        try:
            asg_data = await loop.run_in_executor(None, self.aws.describe_asg)
        except Exception as e:
            logger.error("asg_describe_failed", error=str(e))
            asg_data = None

        try:
            alarms = await loop.run_in_executor(None, self.aws.describe_alarms)
            active_alarms = [a["name"] for a in alarms if a["state"] == "ALARM"]
        except Exception as e:
            logger.warning("alarm_describe_failed", error=str(e))
            active_alarms = []

        metrics_summary = self._build_metrics_summary()

        return {
            "timestamp": datetime.utcnow(),
            "asg_status": asg_data,
            "metrics": metrics_summary,
            "active_alarms": active_alarms,
            "system_healthy": len(active_alarms) == 0 and asg_data is not None,
        }

    def get_recent_data_points(self, hours: int = 2) -> List[Dict]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            p for p in self.buffer.all()
            if p.get("timestamp", datetime.min) >= cutoff
        ]

    def _build_metrics_summary(self) -> List[Dict]:
        buf = self.buffer.all()
        if not buf:
            return []

        now = datetime.utcnow()
        buf_1h = [p for p in buf if p.get("timestamp", now) >= now - timedelta(hours=1)]
        buf_24h = buf

        summaries = []
        for metric in ("cpu_utilization", "memory_utilization"):
            current = buf[-1].get(metric, 0.0) if buf else 0.0
            avg_1h = _mean([p.get(metric, 0.0) for p in buf_1h]) if buf_1h else 0.0
            avg_24h = _mean([p.get(metric, 0.0) for p in buf_24h]) if buf_24h else 0.0
            peak_24h = max((p.get(metric, 0.0) for p in buf_24h), default=0.0)
            trend = _trend(buf_1h, metric)
            summaries.append({
                "metric_name": metric,
                "current_value": round(current, 2),
                "average_1h": round(avg_1h, 2),
                "average_24h": round(avg_24h, 2),
                "peak_24h": round(peak_24h, 2),
                "trend": trend,
            })
        return summaries

    async def _get_instance_count(self) -> int:
        try:
            asg = await asyncio.get_event_loop().run_in_executor(
                None, self.aws.describe_asg
            )
            return asg["current_capacity"] if asg else 1
        except Exception:
            return 1

    def publish_custom_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "Percent",
    ):
        try:
            self.aws.put_metric_data(
                metric_name=metric_name,
                value=value,
                unit=unit,
                dimensions=[{"Name": "AutoScalingGroupName", "Value": self.settings.auto_scaling_group_name}],
            )
        except Exception as e:
            logger.warning("custom_metric_publish_failed", metric=metric_name, error=str(e))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _trend(buf: List[Dict], metric: str) -> str:
    vals = [p.get(metric, 0.0) for p in buf]
    if len(vals) < 4:
        return "stable"
    first_half = _mean(vals[: len(vals) // 2])
    second_half = _mean(vals[len(vals) // 2 :])
    delta = second_half - first_half
    if delta > 5:
        return "rising"
    elif delta < -5:
        return "falling"
    return "stable"
