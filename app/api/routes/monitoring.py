from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, Query
import structlog

from app.models.schemas import MonitoringStatus, AnomalyResult, MetricDataPoint

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/status", response_model=MonitoringStatus)
async def get_monitoring_status(request: Request):
    """Full system health snapshot: ASG state, metrics, and active alarms."""
    monitoring = request.app.state.monitoring
    try:
        status = await monitoring.get_system_status()
        if not status.get("asg_status"):
            raise HTTPException(status_code=503, detail="Could not retrieve ASG status")

        from app.models.schemas import ASGStatus, InstanceInfo, InstanceState, MetricSummary
        asg_data = status["asg_status"]
        instances = [
            InstanceInfo(
                instance_id=i["instance_id"],
                state=_parse_instance_state(i["state"]),
                instance_type=i["instance_type"],
                launch_time=i.get("launch_time"),
                private_ip=i.get("private_ip"),
                public_ip=i.get("public_ip"),
                availability_zone=i.get("availability_zone", ""),
            )
            for i in asg_data.get("instances", [])
        ]
        asg_status = ASGStatus(
            group_name=asg_data["group_name"],
            min_size=asg_data["min_size"],
            max_size=asg_data["max_size"],
            desired_capacity=asg_data["desired_capacity"],
            current_capacity=asg_data["current_capacity"],
            instances=instances,
            health_check_type=asg_data.get("health_check_type", "EC2"),
            created_time=asg_data.get("created_time"),
        )
        metrics = [
            MetricSummary(**m) for m in status.get("metrics", [])
        ]
        return MonitoringStatus(
            timestamp=status["timestamp"],
            asg_status=asg_status,
            metrics=metrics,
            active_alarms=status.get("active_alarms", []),
            system_healthy=status.get("system_healthy", False),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("monitoring_status_error", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/metrics/recent")
async def get_recent_metrics(
    request: Request,
    hours: int = Query(default=2, ge=1, le=168),
):
    """Return raw metric observations from the in-memory buffer."""
    monitoring = request.app.state.monitoring
    points = monitoring.get_recent_data_points(hours=hours)
    return {
        "count": len(points),
        "hours": hours,
        "data": [
            {**p, "timestamp": p["timestamp"].isoformat() if hasattr(p["timestamp"], "isoformat") else p["timestamp"]}
            for p in points
        ],
    }


@router.post("/metrics/ingest")
async def ingest_metric(body: MetricDataPoint, request: Request):
    """
    Push a metric observation into the system (from an external agent,
    custom CloudWatch agent, or test harness).
    """
    monitoring = request.app.state.monitoring
    anomaly_detector = request.app.state.anomaly_detector

    monitoring.ingest_metric(body.model_dump())

    anomaly_result = None
    if anomaly_detector.is_trained:
        raw = anomaly_detector.detect(body)
        anomaly_result = {
            "is_anomaly": raw["is_anomaly"],
            "severity": raw["severity"],
            "anomaly_score": round(raw["anomaly_score"], 4),
            "affected_metrics": raw["affected_metrics"],
        }

    return {
        "ingested": True,
        "timestamp": body.timestamp.isoformat(),
        "anomaly_check": anomaly_result,
    }


@router.get("/alarms")
async def list_alarms(request: Request):
    """List all CloudWatch alarms and their current states."""
    aws = request.app.state.aws
    try:
        alarms = aws.describe_alarms()
        return {"alarms": alarms, "count": len(alarms)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/alarms/{alarm_name}/history")
async def get_alarm_history(alarm_name: str, request: Request, hours: int = 24):
    """Retrieve state-change history for a specific CloudWatch alarm."""
    aws = request.app.state.aws
    try:
        history = aws.get_alarm_history(alarm_name, hours=hours)
        return {"alarm_name": alarm_name, "history": history}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


def _parse_instance_state(raw: str):
    from app.models.schemas import InstanceState
    mapping = {
        "InService": InstanceState.RUNNING,
        "running": InstanceState.RUNNING,
        "pending": InstanceState.PENDING,
        "Pending": InstanceState.PENDING,
        "stopping": InstanceState.STOPPING,
        "stopped": InstanceState.STOPPED,
        "terminated": InstanceState.TERMINATED,
        "Terminating": InstanceState.STOPPING,
        "Terminated": InstanceState.TERMINATED,
    }
    return mapping.get(raw, InstanceState.RUNNING)
